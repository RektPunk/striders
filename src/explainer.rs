use faer::{Col, Mat, prelude::*};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub struct StrideExplainer {
    // Hyperparameters
    m_landmarks: usize,
    lambda: f32,
    sigma: f32,

    // Learned Parameters
    base_value: f32,             // E[f(X)]
    landmarks: Vec<Mat<f32>>,    // samples
    coefficients: Vec<Col<f32>>, // weight w.r.t. samples (Alpha)

    // Projection Components
    whitening_matrices: Vec<Mat<f32>>, // inv_sqrt_eigen
    feature_offsets: Vec<Col<f32>>,    // z_means
    is_fitted: bool,
}

impl StrideExplainer {
    pub fn new(m_landmarks: usize, lambda: f32, sigma: f32) -> Self {
        Self {
            m_landmarks,
            lambda,
            sigma,
            base_value: 0.0,
            landmarks: Vec::new(),
            coefficients: Vec::new(),
            whitening_matrices: Vec::new(),
            feature_offsets: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, x: &Mat<f32>, pred: &Col<f32>) {
        let n = x.nrows();
        let num_features = x.ncols();
        let base_value = pred.iter().sum::<f32>() / n as f32;
        let target_centered = pred - Col::<f32>::full(n, base_value);
        let s2 = 2.0 * self.sigma * self.sigma;

        // Nystrom approximation
        let feature_params: Vec<_> = (0..num_features)
            .into_par_iter()
            .map(|f_idx| {
                let mut rng = rng();
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);
                let landmark_indices = &indices[..self.m_landmarks];

                let mut l_vals = Mat::<f32>::zeros(1, self.m_landmarks);
                for (j_idx, &row_idx) in landmark_indices.iter().enumerate() {
                    l_vals[(0, j_idx)] = x[(row_idx, f_idx)];
                }

                // K_nm (N x m) & K_mm (m x m) 계산
                let mut k_nm = Mat::<f32>::zeros(n, self.m_landmarks);
                let mut k_mm = Mat::<f32>::zeros(self.m_landmarks, self.m_landmarks);

                for i in 0..n {
                    for j in 0..self.m_landmarks {
                        let diff = x[(i, f_idx)] - l_vals[(0, j)];
                        k_nm[(i, j)] = (-(diff * diff) / s2).exp();
                    }
                }
                for i in 0..self.m_landmarks {
                    for j in 0..self.m_landmarks {
                        let diff = l_vals[(0, i)] - l_vals[(0, j)];
                        k_mm[(i, j)] = (-(diff * diff) / s2).exp();
                    }
                }

                let eig = k_mm.self_adjoint_eigen(faer::Side::Lower).unwrap();
                let mut inv_s = Mat::<f32>::zeros(self.m_landmarks, self.m_landmarks);
                for d in 0..self.m_landmarks {
                    let val = eig.S()[d];
                    inv_s[(d, d)] = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                }
                let projection = eig.U() * &inv_s;
                let mut z = &k_nm * &projection;

                let mut z_col_means = Col::<f32>::zeros(self.m_landmarks);
                for j in 0..self.m_landmarks {
                    let m = z.col(j).iter().sum::<f32>() / n as f32;
                    z_col_means[j] = m;
                    for i in 0..n {
                        z[(i, j)] -= m;
                    }
                }

                (l_vals, projection, z, z_col_means)
            })
            .collect();

        // Save for explain
        let mut z_matrices = Vec::with_capacity(num_features);
        for (l, w, z, o) in feature_params {
            self.landmarks.push(l);
            self.whitening_matrices.push(w);
            z_matrices.push(z);
            self.feature_offsets.push(o);
        }

        // Global Ridge Regression
        let total_m = num_features * self.m_landmarks;
        let mut z_total = Mat::<f32>::zeros(n, total_m);
        for (f_idx, z) in z_matrices.iter().enumerate() {
            let offset = f_idx * self.m_landmarks;
            z_total
                .as_mut()
                .submatrix_mut(0, offset, n, self.m_landmarks)
                .copy_from(z);
        }

        let lhs = z_total.transpose() * &z_total;
        let mut ridge_lhs = lhs;
        for i in 0..total_m {
            ridge_lhs[(i, i)] += self.lambda;
        }
        let rhs = z_total.transpose() * &target_centered;
        let alpha_total = ridge_lhs.ldlt(faer::Side::Lower).unwrap().solve(&rhs);

        // Coefficients
        let coefficients: Vec<Col<f32>> = (0..num_features)
            .map(|f_idx| {
                let start = f_idx * self.m_landmarks;
                alpha_total
                    .as_ref()
                    .subrows(start, self.m_landmarks)
                    .to_owned()
            })
            .collect();

        self.base_value = base_value;
        self.coefficients = coefficients;
        self.is_fitted = true;
    }

    pub fn explain(&self, x: &Mat<f32>) -> (Col<f32>, Mat<f32>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let s2 = 2.0 * self.sigma * self.sigma;

        // strides score (N x M)
        let mut all_strides = Mat::<f32>::zeros(n_samples, n_features);

        // Batch processing for each column
        for f_idx in 0..n_features {
            // Batch Kernel: K_batch (N x n_landmarks)
            let mut k_batch = Mat::<f32>::zeros(n_samples, self.m_landmarks);
            let f_landmarks = &self.landmarks[f_idx]; // (1 x n_landmarks)
            for i in 0..n_samples {
                let x_val = x[(i, f_idx)];
                for j in 0..self.m_landmarks {
                    let diff = x_val - f_landmarks[(0, j)];
                    k_batch[(i, j)] = (-(diff * diff) / s2).exp();
                }
            }

            // Batch Projection & Centering
            // Z_batch = K_batch * whitening_matrix - offset
            let mut z_batch = &k_batch * &self.whitening_matrices[f_idx];

            for j in 0..self.m_landmarks {
                let offset = self.feature_offsets[f_idx][j];
                for i in 0..n_samples {
                    z_batch[(i, j)] -= offset;
                }
            }

            // Batch Contribution: stride = Z_batch * coefficient (N x 1) vector
            let stride_col = &z_batch * &self.coefficients[f_idx];
            for i in 0..n_samples {
                all_strides[(i, f_idx)] = stride_col[i];
            }
        }

        // predictions = Base Value + sum of strides
        let mut predictions = Col::<f32>::full(n_samples, self.base_value);
        for i in 0..n_samples {
            let row_sum: f32 = (0..n_features).map(|j| all_strides[(i, j)]).sum();
            predictions[i] += row_sum;
        }

        (predictions, all_strides)
    }
}
