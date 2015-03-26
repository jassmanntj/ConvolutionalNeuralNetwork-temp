package cnn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jblas.DoubleMatrix;

public class ConvolutionLayer extends NeuralNetworkLayer {
	private int channels;
	private int patchDim;
	private int imageRows;
	private int imageCols;
	private int poolDim;
	private int numPatches;
	private double sparsityParam;
	private double lambda;
	private double beta;
	private double alpha;
	private DoubleMatrix whitenedTheta;
	private DoubleMatrix whitenedBias;
	private int patchSize;
	private int imageSize;
	private int resultRows;
	private int resultCols;
	private boolean whiten;
	private static final int NUMTHREADS = 8;
	
	public ConvolutionLayer(int channels, int patchDim, int imageRows, int imageCols, int poolDim, int numPatches, double sparsityParam, double lambda, double beta, double alpha, boolean whiten) {
		this.channels = channels;
		this.patchDim = patchDim;
		this.imageRows = imageRows;
		this.imageCols = imageCols;
		this.poolDim = poolDim;
		this.numPatches = numPatches;
		this.sparsityParam = sparsityParam;
		this.lambda = lambda;
		this.beta = beta;
		this.alpha = alpha;
		this.whiten = whiten;
        this.patchSize = patchDim*patchDim;
	}
	
	public ConvolutionLayer(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] data = reader.readLine().split(",");
			whitenedTheta = new DoubleMatrix(Integer.parseInt(data[0]),Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < whitenedTheta.rows; i++) {
				for(int j = 0; j < whitenedTheta.columns; j++) {
					whitenedTheta.put(i, j, Double.parseDouble(data[i*whitenedTheta.columns+j]));
				}
			}
			data = reader.readLine().split(",");
			whitenedBias = new DoubleMatrix(Integer.parseInt(data[0]), Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					whitenedBias.put(i, j, Double.parseDouble(data[i*whitenedBias.columns+j]));
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
	
	public int getOutputSize() {
		return numPatches * ((imageRows-patchDim+1)/poolDim) * ((imageCols - patchDim+1)/poolDim);
	}
	
	private DoubleMatrix convolve(DoubleMatrix in, DoubleMatrix feat) {
		System.out.println("Starting Convolution");
		int numFeatures = feat.columns;
		this.resultRows = (imageRows-patchDim+1)/poolDim;
		this.resultCols = (imageCols-patchDim+1)/poolDim;
		this.imageSize = imageRows*imageCols;
		DoubleMatrix pooledFeatures = new DoubleMatrix(in.rows, numFeatures * (resultRows * resultCols));
		ExecutorService executor = Executors.newFixedThreadPool(NUMTHREADS);
		for(int imageNum = 0; imageNum < in.rows; imageNum++) {
			Runnable worker = new ConvolutionThread(imageNum, in, feat, pooledFeatures);
			executor.execute(worker);
		}
		executor.shutdown();
		while(!executor.isTerminated());
		return pooledFeatures;
	}

    private DoubleMatrix conv(DoubleMatrix in, DoubleMatrix feat) throws IOException {
        System.out.println("Starting Convolution");
        int resultRows = (imageRows-patchDim+1);
        int resultCols = (imageCols-patchDim+1);
        this.imageSize = imageRows*imageCols;
        DoubleMatrix result = new DoubleMatrix(in.rows, feat.columns * (resultRows * resultCols));
        for(int imageNum = 0; imageNum < in.rows; imageNum++) {
            DoubleMatrix currentImage = null;
            for (int featureNum = 0; featureNum < feat.columns; featureNum++) {
                if(currentImage == null) currentImage = convFeature(in.getRow(imageNum), feat, featureNum);
                else currentImage = DoubleMatrix.concatHorizontally(currentImage, convFeature(in.getRow(imageNum), feat, featureNum));
            }
            result.putRow(imageNum, currentImage);

        }
        return result;
    }
	
	private class ConvolutionThread implements Runnable {
		private int imageNum;
        private DoubleMatrix in;
        private DoubleMatrix feat;
        private DoubleMatrix pooledFeatures;

		public ConvolutionThread(int imageNum, DoubleMatrix in, DoubleMatrix feat, DoubleMatrix pooledFeatures) {
			this.imageNum = imageNum;
            this.in = in;
            this.feat = feat;
            this.pooledFeatures = pooledFeatures;
		}

		@Override
		public void run() {
			System.out.println("Image: " + imageNum);
			DoubleMatrix currentImage = in.getRow(imageNum);
			for (int featureNum = 0; featureNum < feat.columns; featureNum++) {
				DoubleMatrix convolvedFeature = null;
				try {
					convolvedFeature = convFeature(currentImage, feat, featureNum);
				} catch (IOException e) {
					e.printStackTrace();
				}
				pool(convolvedFeature, imageNum, featureNum, pooledFeatures);
			}
		}

	}
	
	public DoubleMatrix convFeature(DoubleMatrix currentImage, DoubleMatrix features, int featureNum) throws IOException {
		DoubleMatrix convolvedFeature = DoubleMatrix.zeros(imageRows-patchDim+1,imageCols - patchDim+1);
		for(int channel = 0; channel < channels; channel++) {
			DoubleMatrix feature = features.getRange(patchSize*channel, patchSize*channel+patchSize,featureNum, featureNum+1);
			feature.reshape(patchDim, patchDim);
			DoubleMatrix image = currentImage.getRange(0, 1, imageSize*channel,imageSize*channel+imageSize);
			image.reshape(imageRows, imageCols);
			DoubleMatrix conv = Utils.conv2d(image, Utils.reverseMatrix(feature), true);
			convolvedFeature.addi(conv);
		}
		return Utils.sigmoid(convolvedFeature.add(whitenedBias.get(featureNum)));
	}
	
	public void pool(DoubleMatrix convolvedFeature, int imageNum, int featureNum, DoubleMatrix pooledFeatures) {
		for(int poolRow = 0; poolRow < resultRows; poolRow++) {
			for(int poolCol = 0; poolCol < resultCols; poolCol++) {
				DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDim, poolRow*poolDim+poolDim, poolCol*poolDim, poolCol*poolDim+poolDim);
				pooledFeatures.put(imageNum, featureNum*resultRows*resultCols+poolRow*resultCols+poolCol, patch.mean());
			}
		}
	}

	public void loadLayer(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] line = reader.readLine().split(",");
			whitenedTheta = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
			line = reader.readLine().split(",");
			for(int i = 0; i < whitenedTheta.rows; i++) {
				for(int j = 0; j < whitenedTheta.columns; j++) {
					whitenedTheta.put(i, j, Double.parseDouble(line[i * whitenedTheta.columns + j]));
				}
			}
			line = reader.readLine().split(",");
            whitenedBias = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
            line = reader.readLine().split(",");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					whitenedBias.put(i, j, Double.parseDouble(line[i * whitenedBias.columns + j]));
				}
			}
			imageRows = Integer.parseInt(reader.readLine());
			imageCols = Integer.parseInt(reader.readLine());
			patchDim = Integer.parseInt(reader.readLine());
			poolDim = Integer.parseInt(reader.readLine());
			reader.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public void writeLayer(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			writer.write(whitenedTheta.rows+","+whitenedTheta.columns+"\n");
			for(int i = 0; i < whitenedTheta.rows; i++) {
				for(int j = 0; j < whitenedTheta.columns; j++) {
					writer.write(whitenedTheta.get(i,j)+",");
				}
			}
			writer.write("\n"+whitenedBias.rows+","+whitenedBias.columns+"\n");
			for(int i = 0; i < whitenedBias.rows; i++) {
				for(int j = 0; j < whitenedBias.columns; j++) {
					writer.write(whitenedBias.get(i,j)+",");
				}
			}
			writer.write("\n"+imageRows+"\n"+imageCols+"\n"+patchDim+"\n"+poolDim);
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public DoubleMatrix compute(DoubleMatrix input) {
		DoubleMatrix pooledFeatures = convolve(input, this.whitenedTheta);
		return pooledFeatures;
	}

	@Override
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) {
		LinearDecoder ae = new LinearDecoder(patchDim, channels, numPatches, sparsityParam, lambda, beta, alpha);
		DoubleMatrix patches = ImageLoader.sample(patchDim, patchDim, 100000, input, imageCols, imageRows, channels);
		if(whiten) {
			patches.divi(patches.max());
			DoubleMatrix meanPatch = patches.columnMeans();
			DoubleMatrix ZCAWhite = Utils.calculateZCAWhite(patches, meanPatch, 0.1);
			patches = Utils.ZCAWhiten(patches, meanPatch, ZCAWhite);
			ae.train(patches, patches, iterations);
			DoubleMatrix previousTheta = ae.getTheta();
			DoubleMatrix previousBias = ae.getBias();
			whitenedTheta = ZCAWhite.mmul(previousTheta);
			whitenedBias = previousBias.sub(meanPatch.mmul(whitenedTheta));
		}
		else {
			ae.train(patches, patches, iterations);
			this.whitenedTheta = ae.getTheta();
			this.whitenedBias = ae.getBias();
		}
		return compute(input);
		
	}
}
