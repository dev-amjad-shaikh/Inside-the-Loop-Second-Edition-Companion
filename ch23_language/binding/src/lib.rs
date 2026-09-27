//! A Rust function exposed to CPython with PyO3 (Chapter 23 boundary experiment).
//! ack(text) parses a JSON tool call and returns a small JSON acknowledgement,
//! the same work as boundary.py's in-process function.
use pyo3::prelude::*;

#[pyfunction]
fn ack(text: &str) -> PyResult<String> {
    let v: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let n = v["payload"].as_str().map(|s| s.chars().count()).unwrap_or(0);
    Ok(format!("{{\"ok\":true,\"n\":{n}}}"))
}

#[pymodule]
fn ch23_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ack, m)?)?;
    Ok(())
}
