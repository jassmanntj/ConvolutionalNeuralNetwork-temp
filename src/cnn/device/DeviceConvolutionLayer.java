package cnn.device;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

public class DeviceConvolutionLayer extends DeviceNeuralNetworkLayer {
	DenseDoubleMatrix2D whitenedTheta;
	DenseDoubleMatrix2D whitenedBias;
	DenseDoubleMatrix2D pooledFeatures;
	DenseDoubleMatrix2D input;
	int imageRows;
	int imageCols;
	int patchDim;
	int poolDim;
	
	public DeviceConvolutionLayer(DeviceNeuralNetworkLayer previousLayer, DenseDoubleMatrix2D zcaWhite, DenseDoubleMatrix2D meanPatch, int patchDim, int poolDim, int imageRows, int imageCols) {
		this.patchDim = patchDim;
		this.poolDim = poolDim;
		this.imageRows = imageRows;
		this.imageCols = imageCols;
		DoubleMatrix2D previousTheta = previousLayer.getTheta();
		whitenedBias = (DenseDoubleMatrix2D) previousLayer.getBias();
		whitenedTheta = new DenseDoubleMatrix2D(zcaWhite.rows(), previousTheta.columns());
		DoubleMatrix2D temp = new DenseDoubleMatrix2D(whitenedBias.rows(), whitenedBias.columns());
		zcaWhite.zMult(previousTheta, whitenedTheta);
		meanPatch.zMult(whitenedTheta, temp);
		whitenedBias.assign(temp, DeviceUtils.sub);
	}
	
	public DeviceConvolutionLayer(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] data = reader.readLine().split(",");
			whitenedTheta = new DenseDoubleMatrix2D(Integer.parseInt(data[0]),Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < whitenedTheta.rows(); i++) {
				for(int j = 0; j < whitenedTheta.columns(); j++) {
					whitenedTheta.set(i, j, Double.parseDouble(data[i*whitenedTheta.columns()+j]));
				}
			}
			data = reader.readLine().split(",");
			whitenedBias = new DenseDoubleMatrix2D(Integer.parseInt(data[0]), Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < whitenedBias.rows(); i++) {
				for(int j = 0; j < whitenedBias.columns(); j++) {
					whitenedBias.set(i, j, Double.parseDouble(data[i*whitenedBias.columns()+j]));
				}
			}
			imageRows = Integer.parseInt(reader.readLine());
			imageCols = Integer.parseInt(reader.readLine());
			patchDim = Integer.parseInt(reader.readLine());
			poolDim = Integer.parseInt(reader.readLine());
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public DoubleMatrix2D compute(DoubleMatrix2D input) {
		int numFeatures = whitenedTheta.columns();
		int resultRows = (imageRows - patchDim+1) / poolDim;
		int resultCols = (imageCols - patchDim+1) / poolDim;
		pooledFeatures = new DenseDoubleMatrix2D(input.rows(), numFeatures * (resultRows * resultCols));
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		for(int imageNum = 0; imageNum < input.rows(); imageNum++) {
			DenseDoubleMatrix2D in = (DenseDoubleMatrix2D) d.subMatrix(input, imageNum, imageNum, 0, input.columns()-1);
			for(int featureNum = 0; featureNum < whitenedTheta.columns(); featureNum++) {
				DoubleMatrix2D convolvedFeature = convFeature(in, featureNum);
				pool(convolvedFeature, featureNum, imageNum, resultRows, resultCols);
			}
			System.out.println("Image: "+imageNum);
		}
		writeTheta("DeviceThetas.test");
		return pooledFeatures;
	}
	
	public DoubleMatrix2D convFeature(DenseDoubleMatrix2D input, int featureNum) {
		DenseDoubleMatrix2D convolvedFeature = new DenseDoubleMatrix2D(imageRows - patchDim + 1, imageCols - patchDim + 1);
		int patchSize = patchDim * patchDim;
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		for(int channel = 0; channel < 3; channel++) {
			DenseDoubleMatrix2D feature = (DenseDoubleMatrix2D) d.subMatrix(whitenedTheta, patchSize*channel, patchSize*channel+patchSize-1, featureNum, featureNum);
			feature = reshape(feature, patchDim, patchDim);
			DenseDoubleMatrix2D in = (DenseDoubleMatrix2D) d.subMatrix(input, 0, 0, imageRows*imageCols*channel, imageRows*imageCols*channel+imageRows*imageCols-1);
			in = reshape(in, imageRows, imageCols);
			DoubleMatrix2D conv = DeviceUtils.conv2d(in, feature);
			convolvedFeature.assign(conv, DoubleFunctions.plus);
		}
		convolvedFeature.assign(DoubleFunctions.plus(whitenedBias.get(0, featureNum)));
		return DeviceUtils.sigmoid(convolvedFeature);
	}
	
	private DenseDoubleMatrix2D reshape(DenseDoubleMatrix2D input, int rows, int cols) {
		DenseDoubleMatrix2D result = new DenseDoubleMatrix2D(rows, cols);
		DoubleMatrix1D in = input.vectorize();
		for(int i = 0; i < cols; i++) {
			for(int j = 0; j < rows; j++) {
				result.set(j, i, in.get(i*rows+j));
			}
		}
		return result;
	}
	
	public void pool(DoubleMatrix2D convolvedFeature, int featureNum, int imageNum, int resultRows, int resultCols) {
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		for(int poolRow = 0; poolRow < resultRows; poolRow++) {
			for(int poolCol = 0; poolCol < resultCols; poolCol++) {
				DoubleMatrix2D patch = d.subMatrix(convolvedFeature, poolRow*poolDim, poolRow*poolDim+poolDim-1, poolCol*poolDim, poolCol*poolDim+poolDim-1);
				pooledFeatures.set(imageNum, featureNum*resultRows*resultCols+poolRow*resultCols+poolCol, patch.zSum()/(patch.size()));
			}
		}
	}
	
	public void writeTheta(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(pooledFeatures.rows()+","+pooledFeatures.columns()+"\n");
			for(int i = 0; i < pooledFeatures.rows(); i++){
				for(int j = 0; j < pooledFeatures.columns(); j++) {
					writer.write(pooledFeatures.get(i,j)+",");
				}
			}
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public DoubleMatrix2D getTheta() {
		return null;
	}

	@Override
	public DoubleMatrix2D getBias() {
		return null;
	}

	@Override
	public void loadTheta(String filename) {
	}

}
