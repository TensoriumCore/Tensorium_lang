module {
  func.func @field_ops(%v: tensorium.field<f64, 1, 0>, %w: tensorium.field<f64, 0, 1>) {
    %c0 = tensorium.const 0.0 : tensorium.field<f64, 0, 0>
    %sum = tensorium.add %c0, %c0 : tensorium.field<f64, 0, 0>
    %diff = tensorium.sub %sum, %c0 : tensorium.field<f64, 0, 0>
    %prod = tensorium.mul %v, %w : tensorium.field<f64, 1, 1>
    %quot = tensorium.div %v, %c0 : tensorium.field<f64, 1, 0>
    %d = tensorium.deriv %v {index = "i"} : tensorium.field<f64, 1, 1>
    %ctr = tensorium.contract %prod : tensorium.field<f64, 0, 0>
    tensorium.dt_assign %v, %diff {indices = []} : tensorium.field<f64, 1, 0>, tensorium.field<f64, 0, 0>
    return
  }
}
