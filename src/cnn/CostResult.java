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
	
	public static double[] getGrads(CostResult[] res) {
		int numElements = 0;
		for(int i = 0; i < res.length; i++) {
			numElements += res[i].thetaGrad.length;
			if(res[i].biasGrad != null) {
				numElements += res[i].biasGrad.length;
			}
		}
		double[] result = new double[numElements];
		int l = 0;
		for(int i = 0; i < res.length; i++) {
			DoubleMatrix thetaGrad = res[i].thetaGrad;
			System.arraycopy(thetaGrad.data, 0, result, l, thetaGrad.data.length);
			l += thetaGrad.data.length;
			thetaGrad = res[i].biasGrad;
			if(thetaGrad != null) {
				System.arraycopy(thetaGrad.data, 0, result, l, thetaGrad.data.length);
				l += thetaGrad.data.length;
			}
		}
		return result;
	}
	
	public static double[] getThetas(NeuralNetworkLayer[] saes, SoftmaxClassifier sc) {
		int numElements = 0;
		for(int i = 0; i < saes.length; i++) {
			numElements += saes[i].getTheta().length;
			numElements += saes[i].getBias().length;
		}
		numElements += sc.getTheta().length;
		double[] thetas = new double[numElements];
		int l = 0;
		for(int i = 0; i < saes.length; i++) {
			DoubleMatrix theta = saes[i].getTheta();
			System.arraycopy(theta.data, 0, thetas, l, theta.data.length);
			l += theta.data.length;
			theta = saes[i].getBias();
			System.arraycopy(theta.data, 0, thetas, l, theta.data.length);
			l += theta.data.length;
		}
		DoubleMatrix theta = sc.getTheta();
		System.arraycopy(theta.data, 0, thetas, l, theta.data.length);		
		return thetas;
	}
	
	public static void setThetas(NeuralNetworkLayer[] saes, SoftmaxClassifier sc, double[] thetas) {
		int l = 0;
		for(int i = 0; i < saes.length; i++) {
			DoubleMatrix theta = saes[i].getTheta();
			System.arraycopy(thetas, l, theta.data, 0, theta.data.length);
			l+= theta.data.length;
			theta = saes[i].getBias();
			System.arraycopy(thetas, l, theta.data, 0, theta.data.length);
			l += theta.data.length;
		}
		DoubleMatrix theta = sc.getTheta();
		System.arraycopy(thetas, l, theta.data, 0, theta.data.length);
	}

}