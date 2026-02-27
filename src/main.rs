use faer::{Col, Mat, prelude::*};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::time::Instant;

use csv::ReaderBuilder;
use std::error::Error;

pub struct StrideExplainer {
    n_samples: usize,
    m_landmarks: usize,
    z_matrices: Vec<Mat<f32>>, // 센터링된 N x m 근사 행렬들
    y_mean: f32,               // 전체 평균 (Intercept)
    target_centered: Col<f32>, // 평균이 제거된 y
}

impl StrideExplainer {
    /// 새로운 STRIDE 설명기 생성 및 적합(Fit)
    pub fn fit(x: &Mat<f32>, y: &Col<f32>, m_landmarks: usize, sigma: f32) -> Self {
        let n = x.nrows();
        let num_features = x.ncols();
        let y_mean = y.iter().sum::<f32>() / n as f32;
        let target_centered = y - Col::<f32>::full(n, y_mean);

        // 1. 각 변수별로 Nystrom 근사 및 센터링 수행
        let z_matrices: Vec<Mat<f32>> = (0..num_features)
            .into_par_iter() // 변수별 계산 병렬화
            .map(|f_idx| {
                let mut rng = rng();
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);
                let landmark_indices = &indices[..m_landmarks];

                // K_nm (N x m) 계산
                let mut k_nm = Mat::<f32>::zeros(n, m_landmarks);
                for i in 0..n {
                    for (j_idx, &j) in landmark_indices.iter().enumerate() {
                        let diff = x[(i, f_idx)] - x[(j, f_idx)];
                        k_nm[(i, j_idx)] = (-(diff * diff) / (2.0 * sigma * sigma)).exp();
                    }
                }

                // K_mm (m x m) 계산
                let mut k_mm = Mat::<f32>::zeros(m_landmarks, m_landmarks);
                for (i_idx, &i) in landmark_indices.iter().enumerate() {
                    for (j_idx, &j) in landmark_indices.iter().enumerate() {
                        let diff = x[(i, f_idx)] - x[(j, f_idx)];
                        k_mm[(i_idx, j_idx)] = (-(diff * diff) / (2.0 * sigma * sigma)).exp();
                    }
                }

                // Z = K_nm * (K_mm)^-1/2
                let eig = k_mm
                    .self_adjoint_eigen(faer::Side::Lower)
                    .expect("Eigenvalue decomposition failed to converge");

                let s = eig.S(); // Eigenvalues (Col 형식)
                let u = eig.U(); // Eigenvectors (Mat 형식)
                let mut inv_sqrt_s = Mat::<f32>::zeros(m_landmarks, m_landmarks);
                for d in 0..m_landmarks {
                    // Diag 타입은 인덱싱으로 대각 성분에 접근 가능합니다.
                    let val = s[d];
                    inv_sqrt_s[(d, d)] = if val > 1e-10 { 1.0 / val.sqrt() } else { 0.0 };
                }
                let mut z = &k_nm * (u * &inv_sqrt_s);

                // --- Centering Step ---
                // 각 컬럼의 평균을 구해서 뺌 (Hilbert Space Projection)
                for j in 0..m_landmarks {
                    let col_mean = z.col(j).iter().sum::<f32>() / n as f32;
                    for i in 0..n {
                        z[(i, j)] -= col_mean;
                    }
                }
                z
            })
            .collect();

        Self {
            n_samples: n,
            m_landmarks,
            z_matrices,
            y_mean,
            target_centered,
        }
    }

    /// 모든 변수의 기여도를 계산 (Global Solve)
    pub fn compute_contributions(&self, lambda: f32) -> Vec<Col<f32>> {
        let n_features = self.z_matrices.len();
        let total_m = n_features * self.m_landmarks;

        // 2. 전체 설계 행렬 Z_total 구성 [Z1, Z2, ..., ZM] (N x total_m)
        let mut z_total = Mat::<f32>::zeros(self.n_samples, total_m);
        for (f_idx, z) in self.z_matrices.iter().enumerate() {
            let offset = f_idx * self.m_landmarks;
            for j in 0..self.m_landmarks {
                for i in 0..self.n_samples {
                    z_total[(i, offset + j)] = z[(i, j)];
                }
            }
        }

        // 3. Ridge Regression 수행: (Z^T Z + lambda*I) alpha = Z^T y
        let zt_z = z_total.transpose() * &z_total;
        let mut lhs = zt_z;
        for i in 0..total_m {
            lhs[(i, i)] += lambda;
        }

        let rhs = z_total.transpose() * &self.target_centered;

        // Cholesky 분해로 alpha 구하기
        let alpha = lhs
            .ldlt(faer::Side::Lower)
            .expect("Cholesky decomposition failed")
            .solve(&rhs);

        // 4. 변수별 기여도 복원 f_i = Z_i * alpha_i
        (0..n_features)
            .map(|f_idx| {
                let offset = f_idx * self.m_landmarks;
                let alpha_i = alpha.get(offset..offset + self.m_landmarks);
                &self.z_matrices[f_idx] * alpha_i
            })
            .collect()
    }
}

// --- CSV 로더 함수 ---
fn load_csv_to_mat(path: &str) -> Mat<f32> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();
    let mut v = Vec::new();
    let mut nrows = 0;

    for result in rdr.records() {
        let record = result.unwrap();
        nrows += 1;
        for field in record.iter() {
            v.push(field.parse::<f32>().unwrap());
        }
    }
    let ncols = v.len() / nrows;

    // faer에서 가장 권장하는 '함수형 생성' 방식입니다.
    // (i, j) 좌표를 받아 벡터 v에서 값을 찾아 매핑합니다.
    Mat::from_fn(nrows, ncols, |i, j| v[i * ncols + j])
}

// 매트릭스 생성이 Column Major 기준이므로 row-major인 CSV는 변환이 필요할 수 있습니다.
// 아래는 안전한 Row-major 로딩 방식입니다.
fn load_csv_to_col(path: &str) -> Col<f32> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();
    let mut v = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        v.push(record[0].parse::<f32>().unwrap());
    }
    let nrows = v.len();
    Col::from_fn(nrows, |i| v[i])
}
fn spearman_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 {
        return 1.0;
    }

    fn get_ranks(data: &[f32]) -> Vec<f32> {
        let n = data.len();
        let mut indexed: Vec<(usize, f32)> = data.iter().cloned().enumerate().collect();
        // 값 기준 정렬
        indexed.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            // 값이 같은 구간(동점자) 찾기
            while j < n && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            // 동점자들에게 평균 순위 부여 (e.g., 1위와 2위가 같으면 둘 다 1.5위)
            let avg_rank = (i + j - 1) as f32 / 2.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        ranks
    }

    let a_ranks = get_ranks(a);
    let b_ranks = get_ranks(b);

    let mut d_squared_sum = 0.0f64; // 정밀도를 위해 f64 사용
    for i in 0..n {
        let diff = (a_ranks[i] - b_ranks[i]) as f64;
        d_squared_sum += diff * diff;
    }

    let nf = n as f64;
    let res = 1.0 - (6.0 * d_squared_sum) / (nf * (nf * nf - 1.0));
    res as f32
}

fn main() {
    println!("📂 데이터 로딩 중...");
    let x = load_csv_to_mat("real_x.csv");
    let y_pred_rf = load_csv_to_col("real_y_pred.csv").to_owned(); // Col 형태로 변환

    println!("🚀 STRIDE 벤치마크 시작 (N={}, M={})", x.nrows(), x.ncols());
    let start_total = Instant::now();

    // 1. Fit (Nystrom + Centering)
    let m_landmarks = 100;
    let sigma = 0.3;
    let lambda = 0.1; // ldlt 안정성을 위해 약간 높게 시작

    let explainer = StrideExplainer::fit(&x, &y_pred_rf, m_landmarks, sigma);

    // 2. Solve (f64 Hybrid 추천)
    let contributions = explainer.compute_contributions(lambda);

    let duration = start_total.elapsed();
    println!("✅ STRIDE 완료: {:.4}s", duration.as_secs_f64());

    // 3. R-squared (Fidelity) 계산
    let n = x.nrows();
    let mut y_hat = Col::<f32>::full(n, explainer.y_mean);
    for contrib in &contributions {
        y_hat = y_hat + contrib;
    }

    let rss: f32 = (0..n).map(|i| (y_pred_rf[i] - y_hat[i]).powi(2)).sum();
    let tss: f32 = (0..n)
        .map(|i| (y_pred_rf[i] - explainer.y_mean).powi(2))
        .sum();
    let r2 = 1.0 - (rss / tss);

    println!("{:-<40}", "");
    println!("📊 Fidelity (R^2 to RF): {:.6}", r2);
    println!("{:-<40}", "");

    // 4. 변수별 중요도 (Global Importance) 출력
    println!("💡 변수별 평균 절대 기여도 (Importance):");
    for (i, contrib) in contributions.iter().enumerate() {
        let avg_imp: f32 = contrib.iter().map(|v| v.abs()).sum::<f32>() / n as f32;
        println!("Feature {:02}: {:.6}", i, avg_imp);
    }
    println!("🔍 TreeSHAP과의 유사도 측정 중...");
    let tree_shap_data = load_csv_to_mat("real_tree_shap.csv");
    let mut correlations = Vec::new();

    for i in 0..x.ncols() {
        let stride_imp: Vec<f32> = (0..n).map(|idx| contributions[i][idx]).collect();

        // TreeSHAP 기여도를 Vec<f32>로 복사
        let tree_col = tree_shap_data.col(i);
        let tree_imp: Vec<f32> = (0..n).map(|idx| tree_col[idx]).collect();
        correlations.push(spearman_correlation(&stride_imp, &tree_imp));
    }

    let avg_spearman = correlations.iter().sum::<f32>() / correlations.len() as f32;
    println!("📈 Average Spearman Correlation: {:.6}", avg_spearman);
}

// fn main() {
//     // --- 설정 (Big Data Scenario) ---
//     let n = 100_000; // 10만 샘플
//     let m_features = 50; // 50개 변수
//     let l_landmarks = 200; // 변수당 랜드마크 수

//     println!(
//         "🚀 벤치마크 시작: N={}, M={}, Landmarks={}",
//         n, m_features, l_landmarks
//     );

//     // 1. 가상 데이터 생성 시간 측정
//     let start_data = Instant::now();
//     let mut x = Mat::<f32>::zeros(n, m_features);
//     let mut y = Col::<f32>::zeros(n);
//     // (데이터 생성 로직 생략 - 이전과 동일하되 루프만 확장)
//     println!("✅ 데이터 생성 완료: {:?}", start_data.elapsed());

//     // 2. STRIDE Fit (Nystrom + Centering) 시간 측정
//     // 이 단계는 변수별로 병렬 처리(Rayon)됩니다.
//     let start_fit = Instant::now();
//     let explainer = StrideExplainer::fit(&x, &y, l_landmarks, 1.0);
//     let fit_duration = start_fit.elapsed();
//     println!("⚡ STRIDE Fit 완료 (병렬 처리): {:?}", fit_duration);

//     // 3. Global Solve (Ridge Regression) 시간 측정
//     // Z_total (100,000 x 10,000) 행렬 연산 구간
//     let start_solve = Instant::now();
//     let contributions = explainer.compute_contributions(1e-4);
//     let solve_duration = start_solve.elapsed();
//     println!("🧠 Global Solve 완료: {:?}", solve_duration);

//     println!("------------------------------------------------------------");
//     println!("총 소요 시간: {:?}", fit_duration + solve_duration);
// }

// fn main() {
//     // 예시 데이터 생성 (N=1000, Features=3)
//     let n = 100000; // 샘플 수
//     let m = 3; // 특성 수

//     // 1. 입력 데이터 X 생성: -3.0 ~ 3.0 사이의 랜덤 값
//     let mut x = Mat::<f32>::zeros(n, m);
//     for i in 0..n {
//         for j in 0..m {
//             x[(i, j)] = (rand::random::<f32>() - 0.5) * 6.0;
//         }
//     }
//     let mut y = Col::<f32>::zeros(n);
//     for i in 0..n {
//         let f1 = x[(i, 0)].sin();
//         let f2 = x[(i, 1)].powi(2);
//         let f3 = x[(i, 2)] * 0.5;
//         y[i] = f1 + f2 + f3;
//     }

//     // STRIDE 실행
//     let explainer = StrideExplainer::fit(&x, &y, 200, 1.0);
//     let contributions = explainer.compute_contributions(1e-4);

//     // 4. 결과 검증 (첫 10개 샘플에 대해 정답과 비교)
//     println!(
//         "{:<10} | {:<10} | {:<10} | {:<10}",
//         "Sample", "True f1", "Pred f1", "Diff"
//     );
//     println!("------------------------------------------------------------");

//     // Centering 때문에 Pred 값은 절대값이 아니라 '평균으로부터의 편차'입니다.
//     // 비교를 위해 정답 f1도 평균을 빼서 비교하거나, 상관계수를 봅니다.
//     for i in 0..10 {
//         let true_f1 = x[(i, 0)].sin();
//         let pred_f1 = contributions[0][i];

//         // 주의: pred_f1은 centering 되어 있어 오프셋이 있을 수 있습니다.
//         // 여기서는 흐름(경향성)이 맞는지 확인합니다.
//         println!(
//             "{:<10} | {:<10.4} | {:<10.4} | {:<10.4}",
//             i,
//             true_f1,
//             pred_f1,
//             (true_f1 - pred_f1).abs()
//         );
//     }

//     let mut rss = 0.0;
//     let mut tss = 0.0;
//     let y_mean = y.iter().sum::<f32>() / n as f32;

//     for i in 0..n {
//         let mut pred_y = explainer.y_mean;
//         for j in 0..m {
//             pred_y += contributions[j][i];
//         }
//         rss += (y[i] - pred_y).powi(2);
//         tss += (y[i] - y_mean).powi(2);
//     }

//     println!("------------------------------------------------------------");
//     println!("R-squared: {:.4}", 1.0 - (rss / tss));
// }
