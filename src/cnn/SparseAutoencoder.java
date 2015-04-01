package cnn;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;

public class SparseAutoencoder implements DiffFunction{
	private static final boolean DEBUG = true;
	private int inputSize;
	private int hiddenSize;
	private int outputSize;
	private int m;
	private double rho;
	private double lambda;
	private double beta;
	private double alpha;
	private DoubleMatrix theta1;
	private DoubleMatrix theta2;
	private DoubleMatrix bias1;
	private DoubleMatrix bias2;
	private DoubleMatrix input;
	private DoubleMatrix output;
	private CostResult[] currentCost;
	private DoubleMatrix biasVelocity;
	private DoubleMatrix thetaVelocity;

	
	public SparseAutoencoder(int inputSize, int hiddenSize, double sparsityParam, double lambda, double beta, double alpha) {
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = inputSize;
		this.lambda = lambda;
		this.beta = beta;
		this.alpha = alpha;
		this.rho = sparsityParam;
		initializeParams();
	}
	
	private void initializeParams() {
		double r = Math.sqrt(6)/Math.sqrt(hiddenSize+inputSize+1);
		theta1 = DoubleMatrix.rand(inputSize, hiddenSize).muli(2*r).subi(r);
		theta2 = DoubleMatrix.rand(hiddenSize, outputSize).muli(2*r).subi(r);
		thetaVelocity = new DoubleMatrix(inputSize, hiddenSize);
		biasVelocity = new DoubleMatrix(1, hiddenSize);
		bias1 = DoubleMatrix.zeros(1, hiddenSize);
		bias2 = DoubleMatrix.zeros(1, outputSize);
	}

	public void gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels, CostResult cr, NeuralNetwork cnn) {
		double epsilon = 0.0001;
		DoubleMatrix biasG = new DoubleMatrix(bias1.rows, bias1.columns);
		for(int i = 0; i < bias1.length; i++) {
			bias1.put(i, bias1.get(i)+epsilon);
			CostResult costPlus = cnn.computeCost(input, labels);
			bias1.put(i, bias1.get(i)-2*epsilon);
			CostResult costMinus = cnn.computeCost(input, labels);
			bias1.put(i, bias1.get(i)+epsilon);
			biasG.put(i, (costPlus.cost-costMinus.cost)/(2*epsilon));
		}
		DoubleMatrix biasA = biasG.add(cr.biasGrad);
		DoubleMatrix biasS = biasG.sub(cr.biasGrad);
		System.out.println("SAE Bias Diff: "+biasS.norm2()/biasA.norm2());

		DoubleMatrix thetaG = new DoubleMatrix(theta1.rows, theta1.columns);
		for(int i = 0; i < theta1.length; i++) {
			theta1.put(i, theta1.get(i)+epsilon);
			CostResult costPlus = cnn.computeCost(input, labels);
			theta1.put(i, theta1.get(i)-2*epsilon);
			CostResult costMinus = cnn.computeCost(input, labels);
			theta1.put(i, theta1.get(i)+epsilon);
			thetaG.put(i, (costPlus.cost-costMinus.cost)/(2*epsilon));
		}
		DoubleMatrix thetaA = thetaG.add(cr.thetaGrad);
		DoubleMatrix thetaS = thetaG.sub(cr.thetaGrad);
		System.out.println("SAE Theta Diff: "+thetaS.norm2()/thetaA.norm2());
	}
	
	private DoubleMatrix computeNumericalGradient(DoubleMatrix input, DoubleMatrix output) {
		double epsilon = .0001;
		DoubleMatrix compiledMatrix = DoubleMatrix.zeros(1, theta1.length+theta2.length+bias1.length+bias2.length);
		for(int i = 0; i < compiledMatrix.length; i++) {
			if(i < theta1.length) {
				int j = i/theta1.columns;
				int k = i%theta1.columns;
				DoubleMatrix thetaPlus = theta1.dup();
				DoubleMatrix thetaMinus = theta1.dup();
				thetaPlus.put(j,k,thetaPlus.get(j,k)+epsilon);
				thetaMinus.put(j,k,thetaMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, output, thetaPlus, theta2, bias1, bias2);
				CostResult[] costMinus = cost(input, output, thetaMinus, theta2, bias1, bias2);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
			else if(i < theta1.length + theta2.length) {
				int j = (i-theta1.length)/theta2.columns;
				int k = (i-theta1.length)%theta2.columns;
				DoubleMatrix thetaPlus = theta2.dup();
				DoubleMatrix thetaMinus = theta2.dup();
				thetaPlus.put(j,k,thetaPlus.get(j,k)+epsilon);
				thetaMinus.put(j,k,thetaMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, output, theta1, thetaPlus, bias1, bias2);
				CostResult[] costMinus = cost(input, output, theta1, thetaMinus, bias1, bias2);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
			else if(i < theta1.length + theta2.length + bias1.length) {
				int j = (i-theta1.length-theta2.length)/bias1.columns;
				int k = (i-theta1.length-theta2.length)%bias1.columns;
				DoubleMatrix biasPlus = bias1.dup();
				DoubleMatrix biasMinus = bias1.dup();
				biasPlus.put(j,k,biasPlus.get(j,k)+epsilon);
				biasMinus.put(j,k,biasMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, output, theta1, theta2, biasPlus, bias2);
				CostResult[] costMinus = cost(input, output, theta1, theta2, biasMinus, bias2);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
			else {
				int j = (i-theta1.length-theta2.length - bias1.length)/bias2.columns;
				int k = (i-theta1.length-theta2.length - bias1.length)%bias2.columns;
				DoubleMatrix biasPlus = bias2.dup();
				DoubleMatrix biasMinus = bias2.dup();
				biasPlus.put(j,k,biasPlus.get(j,k)+epsilon);
				biasMinus.put(j,k,biasMinus.get(j,k)-epsilon);
				CostResult[] costPlus = cost(input, output, theta1, theta2, bias1, biasPlus);
				CostResult[] costMinus = cost(input, output, theta1, theta2, bias1, biasMinus);
				compiledMatrix.put(0,i,(costPlus[1].cost-costMinus[1].cost)/(2*epsilon));
			}
		}
		return compiledMatrix;
	}
	
	protected CostResult[] cost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix theta1, DoubleMatrix theta2, DoubleMatrix bias1, DoubleMatrix bias2) {
		DoubleMatrix[][] result = feedForward(input, theta1, theta2, bias1, bias2);
		DoubleMatrix[] thetaGrad = new DoubleMatrix[2];
		thetaGrad[0] = DoubleMatrix.zeros(theta1.rows, theta1.columns);
		thetaGrad[1] = DoubleMatrix.zeros(theta2.rows, theta2.columns);
		// squared error
		DoubleMatrix cost = result[0][2].sub(output);
		cost.muli(cost);
		double squaredErr = cost.sum() / (2 * m);
		//sparsity
		DoubleMatrix means = result[0][1].columnMeans();
		double klsum = MatrixFunctions.log(means.rdiv(rho)).mul(rho).add(MatrixFunctions.log(means.rsub(1).rdiv(1-rho)).mul(1-rho)).sum();
		double sparsity = klsum * beta;
		//weightDecay
		double weightDecay = theta1.mul(theta1).sum();
		weightDecay += theta2.mul(theta2).sum();
		weightDecay *= lambda/2;
		double costSum = squaredErr + weightDecay + sparsity;
		//delta3
		DoubleMatrix delta3 = result[0][2].sub(output);
		delta3.muli(Utils.sigmoidGradient(result[0][2]));
		//sparsity term
		DoubleMatrix betaTerm = means.rdiv(-rho).add(means.rsub(1).rdiv(1-rho)).mul(beta);
		//delta2
		DoubleMatrix delta2 = delta3.mmul(theta2.transpose());
		delta2.addiRowVector(betaTerm);
		delta2.muli(Utils.sigmoidGradient(result[0][1]));
		//W2grad
		thetaGrad[1] = result[0][1].transpose().mmul(delta3);
		thetaGrad[1].divi(m);
		thetaGrad[1].addi(theta2.mul(lambda));
		//W1grad
		thetaGrad[0] = input.transpose().mmul(delta2);
		thetaGrad[0].divi(m);
		thetaGrad[0].addi(theta1.mul(lambda));
		//b2grad
		DoubleMatrix[] biasGrad = new DoubleMatrix[2];
		biasGrad[1] = delta3.columnMeans();
		//b1grad
		biasGrad[0] = delta2.columnMeans();
		CostResult[] results = new CostResult[2];
		results[0] = new CostResult(0, thetaGrad[0], biasGrad[0], delta2);
		results[1] = new CostResult(costSum, thetaGrad[1], biasGrad[1], delta3);
		return results;
	}
	
	
	public CostResult cost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix delta3) {
		//sparsity term
		//DoubleMatrix betaTerm = means.rdiv(-rho).add(means.rsub(1).rdiv(1-rho)).mul(beta);
		delta3.muli(Utils.sigmoidGradient(output));
		//delta2
		DoubleMatrix delta2 = delta3.mmul(theta1.transpose());
		
		//W1grad
		DoubleMatrix thetaGrad = input.transpose().mmul(delta3);
		thetaGrad.divi(input.rows);
		
		//b1grad
		DoubleMatrix biasGrad = delta3.columnMeans();
		
		return new CostResult(0, thetaGrad, biasGrad, delta2);
	}

	public DoubleMatrix backpropagation(CostResult cr, double momentum, double alpha) {
		biasVelocity.muli(momentum).add(cr.biasGrad.mul(alpha));
		thetaVelocity.muli(momentum).addi(cr.thetaGrad.mul(alpha));
		theta1.subi(thetaVelocity);
		bias1.subi(biasVelocity);

		return cr.delta;
	}
	
	public void lbfgsTrain(DoubleMatrix input, DoubleMatrix output, int iterations) {
		this.input = input;
		this.output = output;
		m = input.rows;
		System.out.println("INPUT Rows: "+input.rows+" Cols: "+input.columns);
		System.out.println("Starting lbfgs.");
		System.out.println("-----");
		QNMinimizer qn = new QNMinimizer(15, true);
		double[] initial = new double[theta1.length+theta2.length+bias1.length+bias2.length];
		System.arraycopy(theta1.data, 0, initial, 0, theta1.data.length);
		System.arraycopy(theta2.data, 0, initial, theta1.data.length, theta2.data.length);
		System.arraycopy(bias1.data, 0, initial, theta1.data.length+theta2.data.length, bias1.data.length);
		System.arraycopy(bias2.data, 0, initial, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data.length);
		initial = qn.minimize(this, 1e-5, initial, iterations);
		System.arraycopy(initial, 0, theta1.data, 0, theta1.data.length);
		System.arraycopy(initial, theta1.data.length, theta2.data, 0, theta2.data.length);
		System.arraycopy(initial, theta1.data.length+theta2.data.length, bias1.data, 0, bias1.data.length);
		System.arraycopy(initial, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data, 0, bias2.data.length);
	}
	
	private DoubleMatrix[][] feedForward(DoubleMatrix patches, DoubleMatrix theta1, DoubleMatrix theta2, DoubleMatrix bias1, DoubleMatrix bias2) {
		DoubleMatrix[][] result = new DoubleMatrix[2][4];
		result[0][0] = patches;
		//z2
		result[1][1] = result[0][0].mmul(theta1);
		DoubleMatrix bias = DoubleMatrix.ones(m,1);
		bias = bias.mmul(bias1);
		result[1][1].addi(bias);
		//a2
		result[0][1] = Utils.sigmoid(result[1][1]);
		//z3
		bias = DoubleMatrix.ones(m, 1);
		bias = bias.mmul(bias2);
		result[1][2] = result[0][1].mmul(theta2);
		result[1][2].addi(bias);
		//a3
		result[0][2] = Utils.sigmoid(result[1][2]);
		return result;
	}
	
	public DoubleMatrix compute(DoubleMatrix input) {
		DoubleMatrix result = input.mmul(theta1);
		result.addiRowVector(bias1);
		return Utils.sigmoid(result);
	}
	
	public void visualize(int size) throws IOException {
		BufferedImage image = new BufferedImage(size*5+12, size*5+12, BufferedImage.TYPE_INT_RGB);
		DoubleMatrix tht1 = theta1.dup();
		tht1.subi(tht1.min());
		tht1.divi(tht1.max());
		tht1.muli(255);
		for(int k = 0; k < 5; k++) {
			for(int l = 0; l < 5; l++) {
				for(int i = 0; i < size; i++) {
					for(int j = 0; j < size; j++) {
						int val = (int)tht1.get(i*size+j, k*5+l);
						int col = (val << 16) | (val << 8) | val;
						image.setRGB(l*10+2+j, k*10+2+i, col);
					}
				}
			}
		}
		File imageFile = new File("Features.png");
		ImageIO.write(image, "png", imageFile);
	}

	@Override
	public int domainDimension() {
		return theta1.length+bias1.length+theta2.length+bias2.length;
	}

	@Override
	public double valueAt(double[] dTheta) {
		return currentCost[1].cost;
	}

	@Override
	public double[] derivativeAt(double[] dTheta) {
		System.arraycopy(dTheta, 0, theta1.data, 0, theta1.data.length);
		System.arraycopy(dTheta, theta1.data.length, theta2.data, 0, theta2.data.length);
		System.arraycopy(dTheta, theta1.data.length+theta2.data.length, bias1.data, 0, bias1.data.length);
		System.arraycopy(dTheta, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data, 0, bias2.data.length);
		currentCost = cost(input, output, theta1, theta2, bias1, bias2);
		double[] derivative = new double[theta1.length+theta2.length+bias1.length+bias2.length];
		System.arraycopy(currentCost[0].thetaGrad.data, 0, derivative, 0, theta1.data.length);
		System.arraycopy(currentCost[1].thetaGrad.data, 0, derivative, theta1.data.length, theta2.data.length);
		System.arraycopy(currentCost[0].biasGrad.data, 0, derivative, theta1.data.length+theta2.data.length, bias1.data.length);
		System.arraycopy(currentCost[1].biasGrad.data, 0, derivative, theta1.data.length+theta2.data.length+bias1.data.length, bias2.data.length);
		return derivative;
	}

	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) throws IOException {
		lbfgsTrain(input, input, iterations);
		return compute(input);
	}
	
	public void writeLayer(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(theta1.rows+","+theta1.columns+"\n");
			for(int i = 0; i < theta1.rows; i++){
				for(int j = 0; j < theta1.columns; j++) {
					writer.write(theta1.get(i,j)+",");
				}
			}
			writer.write("\n"+bias1.rows+","+bias1.columns+"\n");
			for(int i = 0; i < bias1.rows; i++) {
				for(int j = 0; j < bias1.columns; j++) {
					writer.write(bias1.get(i,j)+",");
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
			String[] data = reader.readLine().split(",");
			theta1 = new DoubleMatrix(Integer.parseInt(data[0]),Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < theta1.rows; i++){
				for(int j = 0; j < theta1.columns; j++) {
					theta1.put(i, j, Double.parseDouble(data[i * theta1.columns + j]));
				}
			}
			data = reader.readLine().split(",");
            bias1 = new DoubleMatrix(Integer.parseInt(data[0]),Integer.parseInt(data[1]));
            data = reader.readLine().split(",");
			for(int i = 0; i < bias1.rows; i++) {
				for(int j = 0; j < bias1.columns; j++) {
					bias1.put(i, j, Double.parseDouble(data[i * bias1.columns + j]));
				}
			}
			reader.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
}

