use extendr_api::prelude::*;
use nalgebra::{DMatrix, DVector};

/// Constrói a matriz de base de polinômios (Vandermonde).
fn vandermonde(n: usize, m: usize) -> DMatrix<f64> {
    let mut t = DMatrix::<f64>::zeros(n, m + 1);
    for i in 0..n {
        let x = i as f64 + 1.0;
        let mut p = 1.0;
        for j in 0..=m {
            t[(i, j)] = p;
            p *= x;
        }
    }
    t
}

/// Calcula resíduos de uma regressão por mínimos quadrados.
fn residuals_from_lstsq(t: &DMatrix<f64>, y: &DVector<f64>) -> Result<DVector<f64>> {
    let svd = t.clone().svd(true, true);
    let beta = svd
        .solve(y, 1.0e-14)
        .map_err(|e| Error::Other(format!("Falha na resolução por SVD: {e}")))?;
    let fitted = t * &beta;
    Ok(y - &fitted)
}

/// Função `detrend_cov` interna que opera com tipos Rust nativos.
fn detrend_cov_internal(x: &[f64], y: &[f64], m: i32) -> Result<Vec<f64>> {
    let n = x.len();
    if n < 2 || y.len() != n || m < 0 {
        // Retorna um erro se as condições não forem atendidas.
        return Err(Error::Other(
            "Argumentos inválidos para detrend_cov_internal".into(),
        ));
    }

    let x_vec = DVector::from_vec(x.to_vec());
    let y_vec = DVector::from_vec(y.to_vec());
    let t = vandermonde(n, m as usize);

    let residx = residuals_from_lstsq(&t, &x_vec)?;
    let residy = residuals_from_lstsq(&t, &y_vec)?;

    let denom = (n as f64) - 1.0;
    if denom <= 0.0 {
        return Err(Error::Other("Divisor inválido (n-1).".into()));
    }

    let f2x = residx.norm_squared() / denom;
    let f2y = residy.norm_squared() / denom;
    let f2xy = residx.dot(&residy) / denom;

    Ok(vec![f2x, f2y, f2xy])
}

// logscale computation in pure rust types for internal use
fn logscale_internal(scale_min: i32, scale_max: i32, scale_ratio: f64) -> Result<Vec<i32>> {
    if scale_min <= 0 || scale_max < scale_min || scale_ratio <= 1.0 {
        // Return an empty vector for invalid parameters. In R, this might
        // result in NaN or an error, but returning an empty list is a safe
        // and predictable outcome in Rust.
        let v = Vec::<i32>::new();
        return Ok(v);
    }

    // Use f64 for calculations to maintain precision.
    let scale_min_f = scale_min as f64;
    let scale_max_f = scale_max as f64;

    let r1 = (scale_max_f / scale_min_f).log10();
    let r2 = (scale_ratio).log10();

    let number_of_scales = (r1 / r2).ceil() as usize;
    let mut scales: Vec<f64> = Vec::with_capacity(number_of_scales + 1);
    scales.push(scale_min_f);

    //generate the sequence of scales
    for i in 0..number_of_scales {
        scales.push(scales[i] * scale_ratio);
    }

    // 1. Apply floor to each scale and cast to i32.
    // 2. Sort the vector.
    // 3. Remove consecutive duplicates.
    let mut int_scales: Vec<i32> = scales.iter().map(|&s| s.floor() as i32).collect();
    int_scales.sort_unstable();
    int_scales.dedup();

    // Filter the final vector to ensure no scale exceeds `scale_max`,
    let final_scales: Vec<i32> = int_scales.into_iter().filter(|&s| s <= scale_max).collect();

    Ok(final_scales)
}

// internal dcca computations with rust types
fn dcca_internal(
    x: Vec<f64>,
    y: Vec<f64>,
    order: i32,
    scales: Vec<i32>,
) -> Result<(Vec<i32>, Vec<f64>)> {
    let n = x.len() as usize;

    // --- 1. Criar os perfis (cumsum da série centralizada) ---
    let mean_x: f64 = x.iter().sum::<f64>() / (n as f64);
    let mean_y: f64 = y.iter().sum::<f64>() / (n as f64);

    let mut cumsum_x = 0.0;
    let profile_x: Vec<f64> = x
        .iter()
        .map(|r| {
            cumsum_x += r - mean_x;
            cumsum_x
        })
        .collect();

    let mut cumsum_y = 0.0;
    let profile_y: Vec<f64> = y
        .iter()
        .map(|r| {
            cumsum_y += r - mean_y;
            cumsum_y
        })
        .collect();

    // --- 2. Alocar vetores de resultado ---
    let num_scales = scales.len();
    let mut f2x = vec![0.0; num_scales];
    let mut f2y = vec![0.0; num_scales];
    let mut f2xy = vec![0.0; num_scales];
    let mut rho = vec![0.0; num_scales];

    // --- 3. Loop principal sobre as escalas ---
    for i in 0..num_scales {
        let window = scales[i] as usize;
        if window == 0 {
            continue;
        } // Evita divisão por zero

        let num_blocks = n / window;
        if num_blocks == 0 {
            continue;
        }

        let mut block_f2x = 0.0;
        let mut block_f2y = 0.0;
        let mut block_f2xy = 0.0;

        // --- 4. Loop sobre os blocos ---
        for j in 0..num_blocks {
            let start = j * window;
            let end = start + window;

            let x_slice = &profile_x[start..end];
            let y_slice = &profile_y[start..end];

            // Chama a função interna `detrend_cov` com os slices do perfil
            let var_cov = detrend_cov_internal(x_slice, y_slice, order)?;

            block_f2x += var_cov[0];
            block_f2y += var_cov[1];
            block_f2xy += var_cov[2];
        }

        // --- 5. Normalização e cálculo de rho ---
        // A normalização no código C++ parece atípica. Replicando-a:
        f2x[i] = block_f2x / (n - window) as f64;
        f2y[i] = block_f2y / (n - window) as f64;
        f2xy[i] = block_f2xy / (n - window) as f64;

        let denom_rho = (f2x[i] * f2y[i]).sqrt();
        if denom_rho > 1e-10 {
            // Evita divisão por zero
            rho[i] = f2xy[i] / denom_rho;
        }
    }
    Ok((scales, rho))
}

/// Detrended Covariance
/// Wrapper function for the rust function for doing the detrending.
/// @param x numeric vector.
/// @param y numeric vector.
/// @param m polinimial order for detrending (ex: 1=linear).
/// @return vector with the sum of squared residuals.
///@export
#[extendr]
fn detrend_cov(x: Doubles, y: Doubles, m: i32) -> Result<Robj> {
    let x_f64: Vec<f64> = x.iter().map(|r| r.inner()).collect();
    let y_f64: Vec<f64> = y.iter().map(|r| r.inner()).collect();

    //Validações principais
    if x_f64.is_empty() || y_f64.len() != x_f64.len() {
        return Err(Error::Other(format!(
            "Vectors of incompatible lengths (x={}, y={}).",
            x_f64.len(),
            y_f64.len()
        )));
    }

    let result = detrend_cov_internal(&x_f64, &y_f64, m)?;
    Ok(r!(result))
}

/// Detrended Cross-Correlation Analysis
/// @param x Vetor numérico (série temporal).
/// @param y Vetor numérico (série temporal).
/// @param order Ordem do polinômio para detrending (ex: 1=linear).
/// @param scales Vetor de inteiros com as escalas para o cálculo.
/// @return Uma lista com `scales` e os coeficientes `rho`.
/// @export
#[extendr]
fn dcca(x: Doubles, y: Doubles, order: i32, scales: Integers) -> Result<Robj> {
    let n = x.len();
    if n == 0 || y.len() != n {
        return Err(Error::Other(format!(
            "Vectors of incompatible lengths (x={}, y={}).",
            n,
            y.len()
        )));
    }

    // --- 1. Criar os perfis (cumsum da série centralizada) ---
    // let mean_x = x.iter().map(|r| r.inner()).sum::<f64>() / n as f64;
    // let mean_y = y.iter().map(|r| r.inner()).sum::<f64>() / n as f64;

    // let mut cumsum_x = 0.0;
    // let profile_x: Vec<f64> = x
    //     .iter()
    //     .map(|r| {
    //         cumsum_x += r.inner() - mean_x;
    //         cumsum_x
    //     })
    //     .collect();

    // let mut cumsum_y = 0.0;
    // let profile_y: Vec<f64> = y
    //     .iter()
    //     .map(|r| {
    //         cumsum_y += r.inner() - mean_y;
    //         cumsum_y
    //     })
    //     .collect();

    // // --- 2. Alocar vetores de resultado ---
    // let num_scales = scales.len();
    // let mut f2x = vec![0.0; num_scales];
    // let mut f2y = vec![0.0; num_scales];
    // let mut f2xy = vec![0.0; num_scales];
    // let mut rho = vec![0.0; num_scales];

    // // --- 3. Loop principal sobre as escalas ---
    // for i in 0..num_scales {
    //     let window = scales[i].inner() as usize;
    //     if window == 0 {
    //         continue;
    //     } // Evita divisão por zero

    //     let num_blocks = n / window;
    //     if num_blocks == 0 {
    //         continue;
    //     }

    //     let mut block_f2x = 0.0;
    //     let mut block_f2y = 0.0;
    //     let mut block_f2xy = 0.0;

    //     // --- 4. Loop sobre os blocos ---
    //     for j in 0..num_blocks {
    //         let start = j * window;
    //         let end = start + window;

    //         let x_slice = &profile_x[start..end];
    //         let y_slice = &profile_y[start..end];

    //         // Chama a função interna `detrend_cov` com os slices do perfil
    //         let var_cov = detrend_cov_internal(x_slice, y_slice, order)?;

    //         block_f2x += var_cov[0];
    //         block_f2y += var_cov[1];
    //         block_f2xy += var_cov[2];
    //     }

    //     // --- 5. Normalização e cálculo de rho ---
    //     // A normalização no código C++ parece atípica. Replicando-a:
    //     f2x[i] = block_f2x / (n - window) as f64;
    //     f2y[i] = block_f2y / (n - window) as f64;
    //     f2xy[i] = block_f2xy / (n - window) as f64;

    //     let denom_rho = (f2x[i] * f2y[i]).sqrt();
    //     if denom_rho > 1e-10 {
    //         // Evita divisão por zero
    //         rho[i] = f2xy[i] / denom_rho;
    //     }
    // }

    let x_f64: Vec<f64> = x.iter().map(|r| r.inner()).collect();
    let y_f64: Vec<f64> = y.iter().map(|r| r.inner()).collect();
    let scales_i32: Vec<i32> = scales.iter().map(|r| r.inner()).collect();

    let (scales, rho) = dcca_internal(x_f64, y_f64, order, scales_i32)?;

    Ok(list!(scales = scales, rho = rho).into())
}

/// @export
#[extendr]
fn iafft() {
    println!("Not implemented yet!");
}

/// Create logarithmically spaced scales.
///
/// This generates a vector of scales starting from `scale_min`, where each
/// subsequent scale is the previous one multiplied by `scale_ratio`.
/// The process stops once `scale_max` is exceeded.
///
/// # Arguments
/// * `scale_min` - An integer indicating the minimum scale to be resolved.
/// * `scale_max` - An integer indicating the maximum scale to be resolved.
/// * `scale_ratio` - A float indicating the ratio for successive scales.
///
/// # Returns
/// A `Vec<i32>` of logarithmically spaced scales, sorted and unique.
///
/// # Examples
/// // This would be called from R like so:
/// // scales <- logscale(scale_min = 16, scale_max = 1024, scale_ratio = 2)
/// @export
#[extendr]
fn logscale(scale_min: i32, scale_max: i32, scale_ratio: f64) -> Result<Robj> {
    // if scale_min <= 0 || scale_max < scale_min || scale_ratio <= 1.0 {
    //     // Return an empty vector for invalid parameters. In R, this might
    //     // result in NaN or an error, but returning an empty list is a safe
    //     // and predictable outcome in Rust.
    //     let v = Vec::<i32>::new();
    //     return Ok(v.into());
    // }

    // // Use f64 for calculations to maintain precision.
    // let scale_min_f = scale_min as f64;
    // let scale_max_f = scale_max as f64;

    // let r1 = (scale_max_f / scale_min_f).log10();
    // let r2 = (scale_ratio).log10();

    // let number_of_scales = (r1 / r2).ceil() as usize;
    // let mut scales: Vec<f64> = Vec::with_capacity(number_of_scales + 1);
    // scales.push(scale_min_f);

    // //generate the sequence of scales
    // for i in 0..number_of_scales {
    //     scales.push(scales[i] * scale_ratio);
    // }

    // // 1. Apply floor to each scale and cast to i32.
    // // 2. Sort the vector.
    // // 3. Remove consecutive duplicates.
    // let mut int_scales: Vec<i32> = scales.iter().map(|&s| s.floor() as i32).collect();
    // int_scales.sort_unstable();
    // int_scales.dedup();

    // // Filter the final vector to ensure no scale exceeds `scale_max`,
    // let final_scales: Vec<i32> = int_scales.into_iter().filter(|&s| s <= scale_max).collect();
    let final_scales = logscale_internal(scale_min, scale_max, scale_ratio)?;

    Ok(final_scales.into())
}

extendr_module! {

    mod rhodcca;
    fn detrend_cov;
    fn dcca;
    fn logscale;
    fn iafft;

}
