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
    pub base_value: f32,              // E[f(X)]
    pub feature_bases: Vec<Mat<f32>>, // samples
    pub coefficients: Vec<Col<f32>>,  // weight w.r.t. samples (Alpha)

    // Projection Components
    pub proj_matrices: Vec<Mat<f32>>, // inv_sqrt_eigen
    pub feature_means: Vec<Col<f32>>, // z_means
    pub is_fitted: bool,
}

impl StrideExplainer {
    pub fn new(num_bases: usize, lambda: f32, sigma: f32) -> Self {
        Self {
            num_bases,
            lambda,
            sigma,
            base_value: 0.0,
            feature_bases: Vec::new(),
            coefficients: Vec::new(),
            proj_matrices: Vec::new(),
            feature_means: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, x: MatRef<'_, f32>, pred: ColRef<'_, f32>) {
        let n = x.nrows();
        let num_features = x.ncols();
        let base_value = pred.iter().sum::<f32>() / n as f32;
        let target_centered = pred - Col::<f32>::full(n, base_value);
        let s2_inv = 0.5 / (self.sigma * self.sigma);

        // Nystrom approximation
        let feature_params: Vec<_> = (0..num_features)
            .into_par_iter()
            .map(|f_idx| {
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
                let mut bases = Mat::<f32>::zeros(1, self.num_bases);
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    bases[(0, j_idx)] = x[(row_idx, f_idx)];
                }

                // K_nm (N x m)
                let mut k_nm = Mat::<f32>::zeros(n, self.num_bases);
                for j in 0..self.num_bases {
                    let base_val = bases[(0, j)];
                    for i in 0..n {
                        let x_val = x[(i, f_idx)];
                        if !x_val.is_nan() {
                            let diff = x_val - base_val;
                            k_nm[(i, j)] = (-(diff * diff) * s2_inv).exp();
                        }
                    }
                }

                // K_mm (m x m)
                let mut k_mm = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for j in 0..self.num_bases {
                    for i in 0..=j {
                        let diff = bases[(0, i)] - bases[(0, j)];
                        let val = (-(diff * diff) * s2_inv).exp();
                        k_mm[(i, j)] = val;
                        if i != j {
                            k_mm[(j, i)] = val;
                        }
                    }
                }

                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut inv_s = Mat::<f32>::zeros(self.num_bases, self.num_bases);
                for d in 0..self.num_bases {
                    let val = eig.S()[d];
                    inv_s[(d, d)] = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                }
                let proj_matrix = eig.U() * &inv_s;
                let mut z_features = &k_nm * &proj_matrix;

                let mut z_col_means = Col::<f32>::zeros(self.num_bases);
                for j in 0..self.num_bases {
                    let mean_val = z_features.col(j).iter().sum::<f32>() / n as f32;
                    z_col_means[j] = mean_val;

                    for i in 0..n {
                        if !x[(i, f_idx)].is_nan() {
                            z_features[(i, j)] -= mean_val;
                        } else {
                            z_features[(i, j)] = 0.0;
                        }
                    }
                }
                (bases, proj_matrix, z_features, z_col_means)
            })
            .collect();

        // Save for explain
        let mut z_matrices = Vec::with_capacity(num_features);
        for (b, p, z, o) in feature_params {
            self.feature_bases.push(b);
            self.proj_matrices.push(p);
            z_matrices.push(z);
            self.feature_means.push(o);
        }

        // Global Ridge Regression
        let total_dim = num_features * self.num_bases;
        let mut z_stacked = Mat::<f32>::zeros(n, total_dim);
        for (f_idx, z) in z_matrices.iter().enumerate() {
            let offset = f_idx * self.num_bases;
            z_stacked
                .as_mut()
                .submatrix_mut(0, offset, n, self.num_bases)
                .copy_from(z);
        }

        let lhs = z_stacked.transpose() * &z_stacked;
        let mut ridge_lhs = lhs;
        for i in 0..total_dim {
            ridge_lhs[(i, i)] += self.lambda;
        }
        let rhs = z_stacked.transpose() * &target_centered;
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        // Coefficients
        let coefficients: Vec<Col<f32>> = (0..num_features)
            .map(|f_idx| {
                let start = f_idx * self.num_bases;
                alpha_total
                    .as_ref()
                    .subrows(start, self.num_bases)
                    .to_owned()
            })
            .collect();

        self.base_value = base_value;
        self.coefficients = coefficients;
        self.is_fitted = true;
    }

    pub fn explain(&self, x: MatRef<f32>) -> (Col<f32>, Mat<f32>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let s2 = 2.0 * self.sigma * self.sigma;

        let mut contributions = Mat::<f32>::zeros(n_samples, n_features);
        contributions
            .as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, contribs)| {
                // Batch Kernel: K_batch (N x num_bases)
                let mut k_batch = Mat::<f32>::zeros(n_samples, self.num_bases);
                let bases = &self.feature_bases[f_idx];
                let proj = &self.proj_matrices[f_idx];
                let mean = &self.feature_means[f_idx];
                let coeff = &self.coefficients[f_idx];
                for j in 0..self.num_bases {
                    let base_val = bases[(0, j)];
                    for i in 0..n_samples {
                        let x_val = x[(i, f_idx)];
                        if !x_val.is_nan() {
                            let diff = x_val - base_val;
                            k_batch[(i, j)] = (-(diff * diff) / s2).exp();
                        }
                    }
                }

                // Batch Projection & Centering
                // Z_batch = K_batch * projection_matrix - offset
                let mut z_batch = k_batch * proj;
                for j in 0..self.num_bases {
                    let mean_val = mean[j];
                    for i in 0..n_samples {
                        if !x[(i, f_idx)].is_nan() {
                            z_batch[(i, j)] -= mean_val;
                        } else {
                            z_batch[(i, j)] = 0.0;
                        }
                    }
                }
                // Batch Contribution: stride = Z_batch * coefficient (N x 1) vector
                contribs.col_mut(0).copy_from(&z_batch * coeff);
            });

        // Predictions: // Base Value + sum of strides
        let mut predictions = Col::<f32>::full(n_samples, self.base_value);
        for j in 0..n_features {
            let col = contributions.col(j);
            for i in 0..n_samples {
                predictions[i] += col[i];
            }
        }
        (predictions, contributions)
    }
}
