use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

const EPS: f32 = 1e-7;
const BOUNDARY_EPS: f32 = 1e-5;

pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let batch_size = u.nrows();
    let mut result = Array1::zeros(batch_size);
    let sqrtc = c.sqrt();
    
    result
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, dist)| {
            let u_row = u.row(i);
            let v_row = v.row(i);
            
            let u_norm_sq = u_row.dot(&u_row);
            let v_norm_sq = v_row.dot(&v_row);
            let uv = u_row.dot(&v_row);
            
            let numerator = 2.0 * (u_norm_sq * v_norm_sq - uv * uv);
            let denominator = ((1.0 - c * u_norm_sq) * (1.0 - c * v_norm_sq)).max(EPS);
            let lambda = (numerator / denominator).sqrt();
            let two_minus_lambda_sq = (2.0 - lambda).max(EPS);
            
            *dist = ((2.0 + lambda) / two_minus_lambda_sq).acosh() / sqrtc;
        });
    
    result
}

pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let v_row = v.row(i);
            
            let u_norm_sq = u_row.dot(&u_row);
            let v_norm_sq = v_row.dot(&v_row);
            let u_denom = (1.0 - c * u_norm_sq).max(EPS).sqrt();
            let v_denom = (1.0 - c * v_norm_sq).max(EPS).sqrt();
            
            let mut temp_norm_sq = 0.0;
            for j in 0..dim {
                let temp = u_row[j] / u_denom + v_row[j] / v_denom;
                temp_norm_sq += temp * temp;
                row[j] = temp;
            }
            
            let result_denom = (1.0 + (1.0 + c * temp_norm_sq).sqrt()).max(EPS);
            for j in 0..dim {
                row[j] /= result_denom;
            }
        });
    
    result
}

pub fn klein_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let norm = u_row.dot(&u_row).sqrt().max(EPS);
            let scaled_norm = (norm * r).min(1.0 / c.sqrt() - BOUNDARY_EPS);
            let scale = scaled_norm / norm;
            
            for j in 0..dim {
                row[j] = u_row[j] * scale;
            }
        });
    
    result
}

pub fn klein_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();
    let mut result = Array2::zeros((batch_size, dim));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x_norm_sq = x_row.dot(&x_row);
            let denom = (1.0 + (1.0 - c * x_norm_sq).max(0.0).sqrt()).max(EPS);
            
            for j in 0..dim {
                row[j] = x_row[j] / denom;
            }
        });
    
    result
}

pub fn klein_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();
    let mut result = Array2::zeros((batch_size, dim + 1));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x_norm_sq = x_row.dot(&x_row);
            let x0 = 1.0 / (1.0 - c * x_norm_sq).max(EPS).sqrt();
            
            row[0] = x0;
            for j in 0..dim {
                row[j + 1] = x0 * x_row[j];
            }
        });
    
    result
} 