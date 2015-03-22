package cnn;

import java.io.IOException;

import numerical.LBFGS;

import org.jblas.DoubleMatrix;

import edu.stanford.nlp.optimization.DiffFunction;

public class DeepNN extends NeuralNetworkLayer implements DiffFunction {
	private SparseAutoencoder[] saes;
	private SoftmaxClassifier sc;
	private DoubleMatrix input;
	private DoubleMatrix labels;
	public DeepNN(SparseAutoencoder[] saes, SoftmaxClassifier sc) {
		this.saes = saes;
		this.sc = sc;
	}
	
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix labels, int iterations) throws IOException {
		this.input = input;
		this.labels = labels;
		DoubleMatrix data = input;
		for(int i = 0; i < saes.length; i++) {
			data = saes[i].train(data, data, iterations);
		}
		sc.train(data, labels, iterations);
		double[] thetas = CostResult.getThetas(saes, sc);
		
		//Fine tune
		int[] iprint = {1, 0};
		int[] iflag = {0};
		double[] diag = new double[thetas.length];
		for(int i = 0; i < iterations; i++) {
			CostResult[] result = fineTuneBackprop(input, labels);
			thetas = CostResult.getThetas(saes, sc);
			double[] thetaGrads = CostResult.getGrads(result);
			try {
				LBFGS.lbfgs(thetas.length, 5, thetas, result[result.length-1].cost, thetaGrads, false, diag, iprint, 1e-5, 1e-9, iflag);
			} catch (numerical.LBFGS.ExceptionWithIflag e) {
				e.printStackTrace();
			}
			CostResult.setThetas(saes, sc, thetas);

			if(iflag[0]==0) break;
		}
		
		
		//QNMinimizer qn = new QNMinimizer(15, true);
		//thetas = qn.minimize(this, 1e-5, thetas, iterations);
		CostResult.setThetas(saes, sc, thetas);
		return compute(input);
	}
	
	public double gradientChecking(DoubleMatrix input, DoubleMatrix labels) {
		double[] thetas = CostResult.getThetas(saes, sc);
		double epsilon = 0.0001;
		double[] thetaGrad = new double[thetas.length];
		CostResult[] result = fineTuneBackprop(input, labels);
		double[] calcThetaGrad = CostResult.getGrads(result);
		for(int i = 0; i < thetas.length; i++) {
			//if(i%100 == 0)
				//System.out.println(i+":"+thetas.length);
			thetas[i] = thetas[i] - epsilon;
			CostResult.setThetas(saes, sc, thetas);
			CostResult[] resultMinus = fineTuneBackprop(input, labels);
			
			thetas[i] = thetas[i] + 2 * epsilon;
			CostResult.setThetas(saes, sc, thetas);
			CostResult[] resultPlus = fineTuneBackprop(input, labels);
			thetaGrad[i] = (resultPlus[resultPlus.length-1].cost - resultMinus[resultMinus.length-1].cost)/(2*epsilon);
			
			thetas[i] = thetas[i] - epsilon;
		}
		DoubleMatrix compiledGrads = new DoubleMatrix(1, thetaGrad.length, thetaGrad);
		DoubleMatrix compiledCalcGrads = new DoubleMatrix(1, calcThetaGrad.length, calcThetaGrad);
		DoubleMatrix gradMin = compiledGrads.sub(compiledCalcGrads);
		DoubleMatrix gradAdd = compiledGrads.add(compiledCalcGrads);
		return gradMin.norm2()/gradAdd.norm2();		
	}
	
	private CostResult[] fineTuneBackprop(DoubleMatrix input, DoubleMatrix output) {
		DoubleMatrix[] results = new DoubleMatrix[saes.length+1];
		CostResult[] costResults = new CostResult[saes.length+1];
		results[0] = input;
		for(int i = 0; i < saes.length; i++) {
			results[i+1] = saes[i].compute(results[i]);
		}
		CostResult softMaxCost = sc.stackedCost(results[saes.length], output);
		costResults[saes.length] = softMaxCost; 
		DoubleMatrix lastDelta = softMaxCost.delta;
		DoubleMatrix lastTheta = sc.getTheta();
		for(int i = saes.length-1; i >= 0; i--) {
			CostResult costRes = saes[i].stackedCost(results[i], results[i+1], lastDelta, lastTheta);
			lastDelta = costRes.delta;
			lastTheta = saes[i].getTheta();
			costResults[i] = costRes;
		}
		return costResults;
	}
	
	
	
	public DoubleMatrix compute(DoubleMatrix input) {
		for(int i = 0; i < saes.length; i++) {
			input = saes[i].compute(input);
		}
		return sc.compute(input);
	}

	@Override
	public int domainDimension() {
		double[] thetas = CostResult.getThetas(saes, sc);
		return thetas.length;
	}

	@Override
	public double valueAt(double[] arg0) {
		CostResult[] cost = fineTuneBackprop(input, labels);
		return cost[cost.length-1].cost;
	}

	@Override
	public double[] derivativeAt(double[] arg0) {
		CostResult[] cost = fineTuneBackprop(input, labels);
		return CostResult.getGrads(cost);
	}

	@Override
	public DoubleMatrix getTheta() {
		return null;
	}

	@Override
	public DoubleMatrix getBias() {
		return null;
	}

	@Override
	public void writeLayer(String filename) {
		for(int i = 0; i < saes.length; i++) {
			saes[i].writeLayer(i+filename);
		}
		sc.writeLayer(saes.length+filename);
		
	}

	@Override
	public void loadLayer(String filename) {
		for(int i = 0; i < saes.length; i++) {
			saes[i].loadLayer(i+filename);
		}
		sc.loadLayer(saes.length+filename);
	}

    @Override
    public DoubleMatrix feedForward(DoubleMatrix input) {
        return compute(input);
    }
}