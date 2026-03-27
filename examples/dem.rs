use std::f64::consts::PI;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use clow::prelude::{ClowPointable, ClowSlice, ClowStream};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;

const DIM: usize = 3;

const PTX_SRC: &str = r#"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" __global__ void reset_force(double* force, const unsigned int* n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)n[0] * 3) return;
    force[i] = 0.0;
}

extern "C" __global__ void gravity_force(
    double* force, const double* m, const double* body_type,
    double gx, double gy, double gz, const unsigned int* n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)n[0]) return;
    if (body_type[i] < 0.5) return;
    force[3*i+0] += m[i] * gx;
    force[3*i+1] += m[i] * gy;
    force[3*i+2] += m[i] * gz;
}

extern "C" __global__ void dem_force(
    const double* x,
    const double* u,
    double* force,
    const double* m,
    const double* rad,
    const unsigned int* n,
    const double* kn,
    const double* cor_pp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n[0];
    if (i >= N) return;

    int ki = 3 * i;

    double xi0 = x[ki+0];
    double xi1 = x[ki+1];
    double xi2 = x[ki+2];

    double ui0 = u[ki+0];
    double ui1 = u[ki+1];
    double ui2 = u[ki+2];

    double mi = m[i];
    double ri = rad[i];

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int kj = 3 * j;

        double dx = xi0 - x[kj+0];
        double dy = xi1 - x[kj+1];
        double dz = xi2 - x[kj+2];

        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-24) continue;

        double rij = sqrt(r2);
        double overlap = ri + rad[j] - rij;
        if (overlap <= 0.0) continue;

        double inv_r = 1.0 / rij;
        double nij0 = dx * inv_r;
        double nij1 = dy * inv_r;
        double nij2 = dz * inv_r;

        double dvx = ui0 - u[kj+0];
        double dvy = ui1 - u[kj+1];
        double dvz = ui2 - u[kj+2];

        double vij_dot_n = dvx*nij0 + dvy*nij1 + dvz*nij2;

        double vn0 = vij_dot_n * nij0;
        double vn1 = vij_dot_n * nij1;
        double vn2 = vij_dot_n * nij2;

        double mj = m[j];
        double m_eff;
        if (mi <= 0.0 && mj <= 0.0) continue;
        else if (mi <= 0.0) m_eff = mj;
        else if (mj <= 0.0) m_eff = mi;
        else m_eff = (mi * mj) / (mi + mj);

        double e = fmax(cor_pp[0], 1e-6);
        double loge = log(e);
        double beta = loge / sqrt(loge*loge + M_PI*M_PI);
        double eta_n = -2.0 * beta * sqrt(m_eff * kn[0]);

        double fn_spring = kn[0] * overlap;

        double fnx = fn_spring * nij0 - eta_n * vn0;
        double fny = fn_spring * nij1 - eta_n * vn1;
        double fnz = fn_spring * nij2 - eta_n * vn2;

        force[ki+0] += fnx;
        force[ki+1] += fny;
        force[ki+2] += fnz;
    }
}

extern "C" __global__ void freeze_boundary_particles(
    double* force, double* u, const double* body_type, const unsigned int* n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)n[0]) return;
    if (body_type[i] > 0.5) return;
    force[3*i+0] = 0.0; force[3*i+1] = 0.0; force[3*i+2] = 0.0;
    u[3*i+0] = 0.0; u[3*i+1] = 0.0; u[3*i+2] = 0.0;
}

extern "C" __global__ void integrate(
    double* x, double* u, const double* force,
    const double* m, const double* body_type,
    double dt, const unsigned int* n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)n[0]) return;
    if (body_type[i] < 0.5) return;
    double inv_m = 1.0 / m[i];
    u[3*i+0] += force[3*i+0] * inv_m * dt;
    u[3*i+1] += force[3*i+1] * inv_m * dt;
    u[3*i+2] += force[3*i+2] * inv_m * dt;
    x[3*i+0] += u[3*i+0] * dt;
    x[3*i+1] += u[3*i+1] * dt;
    x[3*i+2] += u[3*i+2] * dt;
}
"#;

pub struct Particles {
    pub n_host: Vec<u32>,
    pub n: ClowSlice<u32>,
    pub x: ClowSlice<f64>,
    pub u: ClowSlice<f64>,
    pub force: ClowSlice<f64>,
    pub m: ClowSlice<f64>,
    pub rad: ClowSlice<f64>,
    pub body_type: ClowSlice<f64>,
    pub stream: Arc<CudaStream>,
}

impl Particles {
    pub fn new(n: u32, stream: Arc<CudaStream>) -> Result<Self, DriverError> {
        let n_host = vec![n];
        let n_dev = stream.clow_clone_htod(n_host.as_slice())?;
        let n_usize = n as usize;
        Ok(Self {
            n_host,
            n: n_dev,
            x: stream.clow_alloc_zeros::<f64>(n_usize * DIM)?,
            u: stream.clow_alloc_zeros::<f64>(n_usize * DIM)?,
            force: stream.clow_alloc_zeros::<f64>(n_usize * DIM)?,
            m: stream.clow_alloc_zeros::<f64>(n_usize)?,
            rad: stream.clow_alloc_zeros::<f64>(n_usize)?,
            body_type: stream.clow_alloc_zeros::<f64>(n_usize)?,
            stream,
        })
    }

    pub fn write_vtk(&self, step: usize) -> std::io::Result<()> {
        use std::fs::{self, File};
        use std::io::Write;

        let x_host = self.stream.clow_clone_dtoh(&self.x).unwrap();
        let u_host = self.stream.clow_clone_dtoh(&self.u).unwrap();
        let f_host = self.stream.clow_clone_dtoh(&self.force).unwrap();
        let m_host = self.stream.clow_clone_dtoh(&self.m).unwrap();
        let r_host = self.stream.clow_clone_dtoh(&self.rad).unwrap();

        let n = x_host.len() / 3;
        fs::create_dir_all("output")?;
        let fname = format!("output/out_{:06}.vtk", step);
        let mut file = File::create(fname)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "DEM particles")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET POLYDATA")?;

        writeln!(file, "POINTS {} double", n)?;
        for i in 0..n {
            let k = 3 * i;
            writeln!(file, "{} {} {}", x_host[k], x_host[k + 1], x_host[k + 2])?;
        }

        writeln!(file, "POINT_DATA {}", n)?;

        writeln!(file, "VECTORS velocity double")?;
        for i in 0..n {
            let k = 3 * i;
            writeln!(file, "{} {} {}", u_host[k], u_host[k + 1], u_host[k + 2])?;
        }

        writeln!(file, "VECTORS force double")?;
        for i in 0..n {
            let k = 3 * i;
            writeln!(file, "{} {} {}", f_host[k], f_host[k + 1], f_host[k + 2])?;
        }

        writeln!(file, "SCALARS mass double 1")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for i in 0..n {
            writeln!(file, "{}", m_host[i])?;
        }

        writeln!(file, "SCALARS radius double 1")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for i in 0..n {
            writeln!(file, "{}", r_host[i])?;
        }

        Ok(())
    }
}

fn main() -> Result<(), DriverError> {
    let ptx = compile_ptx(PTX_SRC).unwrap();
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(ptx)?;
    let k_reset = module.load_function("reset_force")?;
    let k_gravity = module.load_function("gravity_force")?;
    let k_dem = module.load_function("dem_force")?;
    let k_freeze = module.load_function("freeze_boundary_particles")?;
    let k_integrate = module.load_function("integrate")?;

    // --- build tank + particles ---
    let tank_r = 2.0f64;
    let tank_h = 4.0f64;
    let wall_rad = 0.2f64;
    let spacing = 2.0 * wall_rad * 0.75;

    let mut x_host = Vec::<f64>::new();
    let mut u_host = Vec::<f64>::new();
    let mut m_host = Vec::<f64>::new();
    let mut r_host = Vec::<f64>::new();
    let mut bt_host = Vec::<f64>::new();

    // circular floor
    let nr = (tank_r / spacing) as usize;
    for ir in 0..nr {
        let rxy = ir as f64 * spacing;
        let ntheta = ((2.0 * PI * rxy) / spacing).max(1.0) as usize;
        for i in 0..ntheta {
            let th = i as f64 * 2.0 * PI / ntheta as f64;
            x_host.extend_from_slice(&[rxy * th.cos(), rxy * th.sin(), 0.0]);
            u_host.extend_from_slice(&[0.0, 0.0, 0.0]);
            m_host.push(0.0);
            r_host.push(wall_rad);
            bt_host.push(0.0);
        }
    }

    // cylindrical wall
    let nz = (tank_h / spacing) as usize;
    let ntheta = (2.0 * PI * tank_r / spacing) as usize;
    for k in 0..nz {
        let z = k as f64 * spacing;
        for i in 0..ntheta {
            let th = i as f64 * 2.0 * PI / ntheta as f64;
            x_host.extend_from_slice(&[tank_r * th.cos(), tank_r * th.sin(), z]);
            u_host.extend_from_slice(&[0.0, 0.0, 0.0]);
            m_host.push(0.0);
            r_host.push(wall_rad);
            bt_host.push(0.0);
        }
    }

    // falling balls (no overlap)
    let mut rng = rand::thread_rng();
    let n_ball = 300_usize;
    let ball_rad = 0.15f64;
    let min_dist = 2.0 * ball_rad * 1.05;
    let z_min = tank_h * 0.3;
    let z_max = tank_h * 0.7;
    let mut centers: Vec<(f64, f64, f64)> = Vec::new();

    while centers.len() < n_ball {
        let rxy = rng.r#gen::<f64>().sqrt() * (tank_r - 2.0 * ball_rad);
        let th = rng.r#gen::<f64>() * 2.0 * PI;
        let xb = rxy * th.cos();
        let yb = rxy * th.sin();
        let zb = z_min + (z_max - z_min) * rng.r#gen::<f64>();

        let ok = centers.iter().all(|&(xj, yj, zj)| {
            (xb - xj).powi(2) + (yb - yj).powi(2) + (zb - zj).powi(2) >= min_dist * min_dist
        });

        if ok {
            centers.push((xb, yb, zb));
            x_host.extend_from_slice(&[xb, yb, zb]);
            u_host.extend_from_slice(&[0.0, 0.0, 0.0]);
            m_host.push(1.0);
            r_host.push(ball_rad);
            bt_host.push(1.0);
        }
    }

    let n_total = (x_host.len() / 3) as u32;
    let mut particles = Particles::new(n_total, stream.clone())?;

    stream.clow_memcpy_htod(x_host.as_slice(), &mut particles.x)?;
    stream.clow_memcpy_htod(u_host.as_slice(), &mut particles.u)?;
    stream.clow_memcpy_htod(m_host.as_slice(), &mut particles.m)?;
    stream.clow_memcpy_htod(r_host.as_slice(), &mut particles.rad)?;
    stream.clow_memcpy_htod(bt_host.as_slice(), &mut particles.body_type)?;

    // --- simulation params ---
    let kn = 1e4f64;
    let cor_pp = 0.6f64;
    let dt = 1e-4f64;
    let tf = 2.0f64;
    let steps = (tf / dt) as usize;

    let cfg_3n = LaunchConfig::for_num_elems(n_total * 3);
    let cfg_n = LaunchConfig::for_num_elems(n_total);

    let kn_dev = stream.clow_clone_htod(&[kn])?;
    let cor_dev = stream.clow_clone_htod(&[cor_pp])?;

    let n_ptr = particles.n.as_device_ptr();
    let x_ptr = particles.x.as_device_ptr();
    let u_ptr = particles.u.as_device_ptr();
    let force_ptr = particles.force.as_device_ptr();
    let m_ptr = particles.m.as_device_ptr();
    let rad_ptr = particles.rad.as_device_ptr();
    let bt_ptr = particles.body_type.as_device_ptr();
    let kn_ptr = kn_dev.as_device_ptr();
    let cor_ptr = cor_dev.as_device_ptr();

    let pb = ProgressBar::new(steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    for step in 0..steps {
        // reset forces
        {
            let mut launch = stream.launch_builder(&k_reset);
            launch.arg(&force_ptr);
            launch.arg(&n_ptr);
            unsafe { launch.launch(cfg_3n)?; }
        }

        // gravity
        {
            let gx = 0.0f64;
            let gy = 0.0f64;
            let gz = -9.81f64;
            let mut launch = stream.launch_builder(&k_gravity);
            launch.arg(&force_ptr);
            launch.arg(&m_ptr);
            launch.arg(&bt_ptr);
            launch.arg(&gx);
            launch.arg(&gy);
            launch.arg(&gz);
            launch.arg(&n_ptr);
            unsafe { launch.launch(cfg_n)?; }
        }

        // DEM contact forces
        {
            let mut launch = stream.launch_builder(&k_dem);
            launch.arg(&x_ptr);
            launch.arg(&u_ptr);
            launch.arg(&force_ptr);
            launch.arg(&m_ptr);
            launch.arg(&rad_ptr);
            launch.arg(&n_ptr);
            launch.arg(&kn_ptr);
            launch.arg(&cor_ptr);
            unsafe { launch.launch(cfg_n)?; }
        }

        // freeze boundary
        {
            let mut launch = stream.launch_builder(&k_freeze);
            launch.arg(&force_ptr);
            launch.arg(&u_ptr);
            launch.arg(&bt_ptr);
            launch.arg(&n_ptr);
            unsafe { launch.launch(cfg_n)?; }
        }

        // integrate
        {
            let mut launch = stream.launch_builder(&k_integrate);
            launch.arg(&x_ptr);
            launch.arg(&u_ptr);
            launch.arg(&force_ptr);
            launch.arg(&m_ptr);
            launch.arg(&bt_ptr);
            launch.arg(&dt);
            launch.arg(&n_ptr);
            unsafe { launch.launch(cfg_n)?; }
        }

        stream.synchronize()?;

        if step % 50 == 0 {
            particles.write_vtk(step).unwrap();
        }
        pb.inc(1);
    }

    pb.finish_with_message("Simulation done");
    Ok(())
}
