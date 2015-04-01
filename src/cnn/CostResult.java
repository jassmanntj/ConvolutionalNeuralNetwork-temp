package cnn;

import org.jblas.DoubleMatrix;

public class CostResult {
	public double cost;
	public DoubleMatrix delta;
	public DoubleMatrix thetaGrad;
	public DoubleMatrix biasGrad;
	
	public CostResult(double cost, DoubleMatrix thetaGrad, DoubleMatrix biasGrad, DoubleMatrix delta) {
		this.cost = cost;
		this.thetaGrad = thetaGrad;
		this.biasGrad = biasGrad;
		this.delta = delta;
	}

}