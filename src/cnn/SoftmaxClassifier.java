package cnn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;

public class SoftmaxClassifier extends NeuralNetworkLayer implements DiffFunction{
	private static final boolean DEBUG = true;
	private int inputSize;
	private int outputSize;
	private int m;
	private double lambda;
	private DoubleMatrix theta;
	private DoubleMatrix input;
	private DoubleMatrix output;
	
	public SoftmaxClassifier(double lambda) {
		this.lambda = lambda;
	}
	
	public DoubleMatrix getTheta() {
		return theta;
	}
	
	private void initializeParams() {
		double r = Math.sqrt(6)/Math.sqrt(inputSize+1);
		theta = DoubleMatrix.rand(inputSize, outputSize).muli(2 * r).subi(r);
	}
	
	public DoubleMatrix computeNumericalGradient(DoubleMatrix input, DoubleMatrix output) {
		double epsilon = 0.0001;
		DoubleMatrix numGrad = DoubleMatrix.zeros(theta.rows, theta.columns);
		for(int i = 0; i < theta.rows; i++) {
			for(int j = 0; j < theta.columns; j++) {
				DoubleMatrix thetaPlus = theta.dup();
				DoubleMatrix thetaMinus = theta.dup();
				thetaPlus.put(i,j,thetaPlus.get(i,j)+epsilon);
				thetaMinus.put(i,j,thetaMinus.get(i,j)-epsilon);
				CostResult costPlus = cost(input, output, thetaPlus);
				CostResult costMinus = cost(input, output, thetaMinus);
				numGrad.put(i,j,(costPlus.cost-costMinus.cost)/(2*epsilon));
			}
		}
		return numGrad;
	}
	
	public CostResult cost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix theta) {
		DoubleMatrix res1 = input.mmul(theta);
		DoubleMatrix maxes = res1.rowMaxs();
		DoubleMatrix res = res1.subColumnVector(maxes);
		MatrixFunctions.expi(res);
		res.diviColumnVector(res.rowSums());
		DoubleMatrix thetaGrad = res.sub(output);
		thetaGrad = input.transpose().mmul(thetaGrad);
		thetaGrad.divi(m);
		thetaGrad.addi(theta.mul(lambda));
		MatrixFunctions.logi(res);
		
		double cost = -res.mul(output).sum()/m + theta.mul(theta).sum() * lambda / 2;
		return new CostResult(cost, thetaGrad, null, null);
	}
	
	public CostResult cost(DoubleMatrix input, DoubleMatrix output) {
		return cost(input,output,theta);
	}
	
	public CostResult stackedCost(DoubleMatrix input, DoubleMatrix output) {
		m = input.rows;
		DoubleMatrix res = input.mmul(theta);
		DoubleMatrix p = res.subColumnVector(res.rowMaxs());
		MatrixFunctions.expi(p);
		p.diviColumnVector(p.rowSums());
		DoubleMatrix thetaGrad =input.transpose().mmul(p.sub(output)).div(m).add(theta.mul(lambda));
		DoubleMatrix delta = p.sub(output).mmul(theta.transpose()).mul(Utils.sigmoidGradient(input));
		MatrixFunctions.logi(p);
		double cost = -p.mul(output).sum()/m + theta.mul(theta).sum()*lambda/2;
		return new CostResult(cost, thetaGrad, null, delta);
	}

	public void gradientDescent(DoubleMatrix input, DoubleMatrix output, int iterations, double alpha) {
		m = input.rows;
		System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
		initializeParams();
		System.out.println("Starting gradient descent with " + iterations + " iterations.");
		System.out.println("-----");
		for(int i = 0; i < iterations; i++) {
			CostResult result = cost(input, output, theta);
			if(DEBUG) {
				DoubleMatrix numGrad = computeNumericalGradient(input, output);
				DoubleMatrix gradMin = numGrad.dup();
				DoubleMatrix gradAdd = numGrad.dup();
				gradMin.subi(result.thetaGrad);
				gradAdd.addi(result.thetaGrad);
				System.out.println("Diff: "+gradMin.norm2()/gradAdd.norm2());
			}
			System.out.println("Interation " + i + " Cost: " + result.cost);
			theta.subi(result.thetaGrad.mul(alpha));
		}
	}
	
	public void lbfgsTrain(DoubleMatrix input, DoubleMatrix output, int iterations) {
		this.input = input;
		this.output = output;
		m = input.rows;
		System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
		initializeParams();
		System.out.println("Starting lbfgs.");
		System.out.println("-----");
		QNMinimizer qn = new QNMinimizer(25, true);
		theta.data = qn.minimize(this, 1e-9, theta.data, iterations);
	}
	
	public void writeTheta(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(theta.rows+","+theta.columns+"\n");
			for(int i = 0; i < theta.length; i++){
				if( i < theta.length-1)
					writer.write(theta.data[i]+",");
				else writer.write(""+theta.data[i]);
			}
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public void writeLayer(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(theta.rows+","+theta.columns+"\n");
			for(int i = 0; i < theta.rows; i++){
				for(int j = 0; j < theta.columns; j++) {
					writer.write(theta.get(i,j)+",");
				}
			}
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

	public void loadLayer(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] thetaSize = reader.readLine().split(",");
			theta = new DoubleMatrix(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			assert data.length == theta.data.length;
			for(int i = 0; i < theta.rows; i++) {
				for(int j = 0; j < theta.columns; j++) {
					theta.put(i, j, Double.parseDouble(data[i * theta.columns + j]));
				}
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public DoubleMatrix loadTheta(String filename, DoubleMatrix input) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] thetaSize = reader.readLine().split(",");
			theta = new DoubleMatrix(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			assert data.length == theta.data.length;
			for(int i = 0; i < theta.data.length; i++) {
				theta.data[i] = Double.parseDouble(data[i]);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		return compute(input);
	}
	
	public void loadTheta(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] thetaSize = reader.readLine().split(",");
			theta = new DoubleMatrix(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			assert data.length == theta.data.length;
			for(int i = 0; i < theta.data.length; i++) {
				theta.data[i] = Double.parseDouble(data[i]);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public int[] computeResults(DoubleMatrix input) {
		DoubleMatrix result = input.mmul(theta);
		int[] results = new int[result.rows];
		for(int i = 0; i < result.rows; i++) {
			double currentMax = 0;
			for(int j = 0; j < result.columns; j++) {
				if(result.get(i,j) > currentMax) {
					currentMax = result.get(i,j);
					results[i] = j;
				}
			}
		}
		return results;
	}
	
	@Override
	public int domainDimension() {
		return theta.length;
	}

	@Override
	public double valueAt(double[] arg0) {
		theta.data = arg0;
		CostResult res = cost(input, output);
		return res.cost;
	}

	@Override
	public double[] derivativeAt(double[] arg0) {
		theta.data = arg0;
		CostResult res = cost(input, output);
		return res.thetaGrad.data;
	}

	@Override
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) {
		inputSize = input.columns;
		outputSize = output.columns;
		initializeParams();
		lbfgsTrain(input, output, iterations);
		return compute(input);
		
	}

	@Override
	public DoubleMatrix compute(DoubleMatrix input) {
		return input.mmul(theta);
	}

	@Override
	public DoubleMatrix getBias() {
		return null;
	}
}
