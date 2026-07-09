use faer::prelude::Solve;
use faer::{Col, ColRef, Mat, MatRef};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub struct StrideExplainer {
    // Hyperparameters
    pub num_bases: usize,
    pub lambda: f32,
    pub sigma: f32,

    // Learned Parameters
    pub base_value: f32,
    pub bases: Mat<f32>,
    pub weights: Mat<f32>,
    pub offsets: Col<f32>,
}

#[inline]
fn rbf_kernel(diff: f32, s2_inv: f32) -> f32 {
    (-(diff * diff) * s2_inv).exp()
}

#[inline]
fn evaluate_feature(
    x: f32,
    bases: ColRef<'_, f32>,
    weights: ColRef<'_, f32>,
    offset: f32,
    s2_inv: f32,
) -> f32 {
    // Computes the contribution of a single feature:
    // f_j(x) = k(x, B_j)^T w_j - b_j
    let mut sum = 0.0;
    for (base, weight) in bases.iter().zip(weights.iter()) {
        sum += rbf_kernel(x - *base, s2_inv) * *weight;
    }

    sum - offset
}

impl StrideExplainer {
    pub fn new(num_bases: usize, lambda: f32, sigma: f32) -> Self {
        Self {
            num_bases,
            lambda,
            sigma,
            base_value: 0.0,
            bases: Mat::zeros(0, 0),
            weights: Mat::zeros(0, 0),
            offsets: Col::zeros(0),
        }
    }

    #[inline]
    fn kernel_scale(&self) -> f32 {
        0.5 / (self.sigma * self.sigma)
    }

    pub fn fit(&mut self, x: MatRef<'_, f32>, pred: ColRef<'_, f32>) {
        let n = x.nrows();
        let num_features = x.ncols();
        let base_value = pred.iter().sum::<f32>() / n as f32;
        let target_centered = pred - Col::<f32>::full(n, base_value);
        let s2_inv = self.kernel_scale();

        let total_dim = num_features * self.num_bases;
        let mut z_stacked = Mat::<f32>::zeros(n, total_dim);
        self.bases = Mat::<f32>::zeros(self.num_bases, num_features);
        let mut means = Mat::<f32>::zeros(self.num_bases, num_features);

        // Nystrom approximation
        let projections: Vec<Mat<f32>> = z_stacked
            .par_col_partition_mut(num_features)
            .zip(self.bases.par_col_partition_mut(num_features))
            .zip(means.par_col_partition_mut(num_features))
            .enumerate()
            .map(|(f_idx, ((mut z_block, base_col), mean_col))| {
                let mut rng = rng();
                let mut valid_indices: Vec<usize> =
                    (0..n).filter(|&i| !x[(i, f_idx)].is_nan()).collect();

                if valid_indices.is_empty() {
                    panic!(
                        "Feature at index {} consists entirely of NaNs. \
                        Please remove constant or empty columns before fitting.",
                        f_idx
                    );
                }

                valid_indices.shuffle(&mut rng);
                let landmark_indices = &valid_indices[..self.num_bases.min(valid_indices.len())];

                let mut basis = base_col.col_mut(0);
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    basis[j_idx] = x[(row_idx, f_idx)];
                }

                // K_nm (N x m)
                let mut k_nm = Mat::<f32>::zeros(n, self.num_bases);
                for j in 0..self.num_bases {
                    let base_val = basis[j];
                    for i in 0..n {
                        let x_val = x[(i, f_idx)];
                        if !x_val.is_nan() {
                            let diff = x_val - base_val;
                            k_nm[(i, j)] = rbf_kernel(diff, s2_inv);
                        }
                    }
                }

                // K_mm (m x m)
                let mut k_mm = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for j in 0..self.num_bases {
                    for i in 0..=j {
                        let diff = basis[i] - basis[j];
                        let val = rbf_kernel(diff, s2_inv);
                        k_mm[(i, j)] = val;
                        if i != j {
                            k_mm[(j, i)] = val;
                        }
                    }
                }

                // Compute the projection matrix from the eigen decomposition of K_mm.
                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut inv_s = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for d in 0..self.num_bases {
                    let val = eig.S()[d];
                    inv_s[(d, d)] = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                }
                let projection = eig.U() * &inv_s;
                let mut z_features = &k_nm * &projection;

                // Center the z_features around zero.
                let mut mean = mean_col.col_mut(0);
                for j in 0..self.num_bases {
                    let mean_val = z_features.col(j).iter().sum::<f32>() / n as f32;
                    mean[j] = mean_val;

                    for i in 0..n {
                        if !x[(i, f_idx)].is_nan() {
                            z_features[(i, j)] -= mean_val;
                        } else {
                            z_features[(i, j)] = 0.0;
                        }
                    }
                }
                z_block.copy_from(&z_features);

                projection
            })
            .collect();

        // Global Ridge Regression
        let mut ridge_lhs = z_stacked.transpose() * &z_stacked;
        for i in 0..total_dim {
            ridge_lhs[(i, i)] += self.lambda;
        }
        let rhs = z_stacked.transpose() * &target_centered;
        let coefficient_stack = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        // Convert the learned coefficients back to kernel-space weights and centering offsets.
        self.weights = Mat::zeros(self.num_bases, num_features);
        self.offsets = Col::zeros(num_features);

        for (f_idx, (projection, mean)) in projections.iter().zip(means.col_iter()).enumerate() {
            let start = f_idx * self.num_bases;
            let coefficient = coefficient_stack.as_ref().subrows(start, self.num_bases);

            // w_j = P_j coefficient
            let weight = projection * coefficient;
            self.weights.col_mut(f_idx).copy_from(weight);

            // b_j = mu_j^T coefficient
            self.offsets[f_idx] = mean.transpose() * coefficient;
        }
        self.base_value = base_value;
    }

    pub fn predict(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let s2_inv = self.kernel_scale();

        let mut predictions = Col::<f32>::full(n_samples, self.base_value);
        predictions
            .as_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, pred_val)| {
                let mut sample_sum = 0.0;
                for f_idx in 0..n_features {
                    let basis = self.bases.col(f_idx);
                    let weight = self.weights.col(f_idx);
                    let offset = self.offsets[f_idx];
                    let x_val = x[(i, f_idx)];
                    if !x_val.is_nan() {
                        sample_sum += evaluate_feature(x_val, basis, weight, offset, s2_inv);
                    }
                }
                *pred_val += sample_sum;
            });

        predictions
    }

    pub fn explain(&self, x: MatRef<f32>) -> Mat<f32> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let s2_inv = self.kernel_scale();

        let mut contributions = Mat::<f32>::zeros(n_samples, n_features);
        contributions
            .as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, mut contribs)| {
                let basis = self.bases.col(f_idx);
                let weight = self.weights.col(f_idx);
                let offset = self.offsets[f_idx];
                for i in 0..n_samples {
                    let x_val = x[(i, f_idx)];
                    if x_val.is_nan() {
                        contribs[(i, 0)] = 0.0;
                    } else {
                        contribs[(i, 0)] = evaluate_feature(x_val, basis, weight, offset, s2_inv);
                    }
                }
            });

        contributions
    }
}
