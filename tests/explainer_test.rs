use faer::{Col, Mat};

pub use striders::explainer::StrideExplainer;

#[test]
fn test_stride_explainer() {
    let n_samples = 400;
    let n_features = 3;
    let num_bases = 32;
    let lambda: f32 = 0.01;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);

    for r_idx in 0..n_samples {
        let v1 = r_idx as f32 * 0.05;
        let v2 = (r_idx as f32 * 0.2).sin();
        let v3 = (r_idx as f32).powi(2) / 5000.0;

        x[(r_idx, 0)] = v1;
        x[(r_idx, 1)] = if r_idx % 10 == 0 { f32::NAN } else { v2 };
        x[(r_idx, 2)] = v3;
        let v2_clean = if v2.is_nan() { 0.0 } else { v2 };
        y[r_idx] = 3.5 * v1 - 2.0 * v2_clean + 0.8 * v3 + 10.0;
    }

    let mut explainer = StrideExplainer::new(num_bases, lambda);
    explainer.fit(x.as_ref(), y.as_ref());

    let y_pred = explainer.predict(x.as_ref());
    let contributions = explainer.explain(x.as_ref());

    assert_eq!(
        contributions.nrows(),
        n_samples,
        "Rows mismatch in contributions"
    );
    assert_eq!(
        contributions.ncols(),
        n_features,
        "Columns mismatch in contributions"
    );
    assert_eq!(y_pred.nrows(), n_samples, "Rows mismatch in predictions");

    for r_idx in 0..n_samples {
        let mut row_contribution_sum = 0.0;
        for f_idx in 0..n_features {
            row_contribution_sum += contributions[(r_idx, f_idx)];
        }

        let reconstructed_pred = explainer.base_value + row_contribution_sum;

        assert!(
            (reconstructed_pred - y_pred[r_idx]).abs() < 1e-4,
            "Sample {}: Consistency check failed. Reconstructed: {:.6}, Pred: {:.6}",
            r_idx,
            reconstructed_pred,
            y_pred[r_idx]
        );

        assert!(
            !y_pred[r_idx].is_nan(),
            "Prediction at index {} is NaN",
            r_idx
        );
    }
}

#[test]
#[should_panic(expected = "Feature 1 has only 0 valid samples but num_bases=16")]
fn test_explain_all_nans_panic() {
    let n_samples = 50;
    let n_features = 2;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let y = Col::<f32>::zeros(n_samples);

    for r_idx in 0..n_samples {
        x[(r_idx, 0)] = r_idx as f32;
        x[(r_idx, 1)] = f32::NAN;
    }

    let mut explainer = StrideExplainer::new(16, 0.01);
    explainer.fit(x.as_ref(), y.as_ref());
}

#[test]
fn test_explain_single_sample() {
    let n_samples = 100;
    let n_features = 2;

    let mut x_train = Mat::<f32>::zeros(n_samples, n_features);
    let mut y_train = Col::<f32>::zeros(n_samples);

    for r_idx in 0..n_samples {
        x_train[(r_idx, 0)] = r_idx as f32 * 0.1;
        x_train[(r_idx, 1)] = (r_idx as f32).cos();
        y_train[r_idx] = x_train[(r_idx, 0)] + x_train[(r_idx, 1)];
    }

    let mut explainer = StrideExplainer::new(16, 0.01);
    explainer.fit(x_train.as_ref(), y_train.as_ref());

    let mut x_single = Mat::<f32>::zeros(1, n_features);
    x_single[(0, 0)] = 5.5;
    x_single[(0, 1)] = -0.3;

    let y_pred = explainer.predict(x_single.as_ref());
    let contributions = explainer.explain(x_single.as_ref());

    assert_eq!(
        y_pred.nrows(),
        1,
        "Single sample prediction must have exactly 1 row"
    );
    assert_eq!(
        contributions.nrows(),
        1,
        "Single sample contributions must have exactly 1 row"
    );
    assert_eq!(
        contributions.ncols(),
        n_features,
        "Single sample contributions must match feature count"
    );

    let recon = explainer.base_value + contributions[(0, 0)] + contributions[(0, 1)];
    assert!(
        (recon - y_pred[0]).abs() < 1e-5,
        "Single sample consistency failed"
    );
}
