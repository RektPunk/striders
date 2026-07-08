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
    pub feature_bases: Vec<Vec<f32>>,
    pub feature_weights: Vec<Vec<f32>>, // W = P * alpha
    pub feature_offsets: Vec<f32>,      // C = mean * alpha
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
            feature_weights: Vec::new(),
            feature_offsets: Vec::new(),
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

                let mut bases = vec![0.0; self.num_bases];
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    bases[j_idx] = x[(row_idx, f_idx)];
                }

                // K_nm (N x m)
                let mut k_nm = Mat::<f32>::zeros(n, self.num_bases);
                for j in 0..self.num_bases {
                    let base_val = bases[j];
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
                        let diff = bases[i] - bases[j];
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

        // Global Ridge Regression
        let total_dim = num_features * self.num_bases;
        let mut z_stacked = Mat::<f32>::zeros(n, total_dim);
        for (f_idx, (_, _, z, _)) in feature_params.iter().enumerate() {
            let offset = f_idx * self.num_bases;
            z_stacked
                .as_mut()
                .submatrix_mut(0, offset, n, self.num_bases)
                .copy_from(z);
        }

        let mut ridge_lhs = z_stacked.transpose() * &z_stacked;
        for i in 0..total_dim {
            ridge_lhs[(i, i)] += self.lambda;
        }
        let rhs = z_stacked.transpose() * &target_centered;
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        self.feature_bases = Vec::with_capacity(num_features);
        self.feature_weights = Vec::with_capacity(num_features);
        self.feature_offsets = Vec::with_capacity(num_features);

        for f_idx in 0..num_features {
            let start = f_idx * self.num_bases;
            let coeff = alpha_total.as_ref().subrows(start, self.num_bases);
            let (bases, proj, _, mean) = &feature_params[f_idx];

            let w_col = proj * coeff; // W = P * alpha
            let w_vec: Vec<f32> = (0..self.num_bases).map(|j| w_col[j]).collect();

            let mut c = 0.0;
            for j in 0..self.num_bases {
                c += mean[j] * coeff[j]; // C = mean * alpha
            }

            self.feature_bases.push(bases.to_owned());
            self.feature_weights.push(w_vec);
            self.feature_offsets.push(c);
        }

        self.base_value = base_value;
        self.is_fitted = true;
    }

    pub fn predict(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let s2_inv = 0.5 / (self.sigma * self.sigma);

        let mut predictions = Col::<f32>::full(n_samples, self.base_value);
        predictions
            .as_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, pred_val)| {
                let mut sample_sum = 0.0;
                for f_idx in 0..n_features {
                    let x_val = x[(i, f_idx)];
                    if !x_val.is_nan() {
                        let bases = &self.feature_bases[f_idx];
                        let weights = &self.feature_weights[f_idx];
                        let offset = self.feature_offsets[f_idx];

                        let mut sum = 0.0;
                        for j in 0..self.num_bases {
                            let diff = x_val - bases[j];
                            sum += (-(diff * diff) * s2_inv).exp() * weights[j];
                        }
                        sample_sum += sum - offset;
                    }
                }
                *pred_val += sample_sum;
            });

        predictions
    }

    pub fn explain(&self, x: MatRef<f32>) -> Mat<f32> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let s2_inv = 0.5 / (self.sigma * self.sigma);

        let mut contributions = Mat::<f32>::zeros(n_samples, n_features);
        contributions
            .as_mut()
            .par_col_partition_mut(n_features)
            .enumerate()
            .for_each(|(f_idx, mut contribs)| {
                let bases = &self.feature_bases[f_idx];
                let weights = &self.feature_weights[f_idx];
                let offset = self.feature_offsets[f_idx];

                for i in 0..n_samples {
                    let x_val = x[(i, f_idx)];
                    if x_val.is_nan() {
                        contribs[(i, 0)] = 0.0;
                    } else {
                        let mut sum = 0.0;
                        for j in 0..self.num_bases {
                            let diff = x_val - bases[j];
                            sum += (-(diff * diff) * s2_inv).exp() * weights[j];
                        }
                        contribs[(i, 0)] = sum - offset;
                    }
                }
            });

        contributions
    }
}
