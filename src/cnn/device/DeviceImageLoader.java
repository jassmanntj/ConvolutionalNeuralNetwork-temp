package cnn.device;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

import javax.imageio.ImageIO;

import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;

import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

public class DeviceImageLoader {
	DenseDoubleMatrix2D images;
	DenseDoubleMatrix2D labels;
	int channels;
	int width, height;
	HashMap<String, Double> labelMap;
	DoubleFactory2D f = DoubleFactory2D.dense;
	public void loadFolder(File folder, int channels, int width, int height) throws IOException {
		labelMap = new HashMap<String, Double>();
		this.channels = channels;
		this.width = width;
		this.height = height;
		double labelNo = -1;
		for(File leaf : folder.listFiles()) {
			leaf.listFiles();
			if(leaf.isDirectory()) {
				for(File photos : leaf.listFiles()) {
					if(photos.getName().equals("photograph")) {
						labelMap.put(leaf.getName(), ++labelNo);
						for(File image : photos.listFiles()) {
							BufferedImage img = ImageIO.read(image);
							int[] pixels = new int[1];
							if(img.getHeight()==600 && img.getWidth() == 800) {
								img = Scalr.resize(img, Method.QUALITY, height, width);
								img = Scalr.rotate(img, Rotation.CW_90);
								pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
							}
							else if(img.getHeight()==800 && img.getWidth()==600) {
								img = Scalr.resize(img, Method.QUALITY, width, height);
								pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
							}
							img.flush();
							if(pixels.length==width*height) {
								DenseDoubleMatrix2D row = new DenseDoubleMatrix2D(1, pixels.length*channels);
								DenseDoubleMatrix2D lRow = new DenseDoubleMatrix2D(1,1);
								lRow.set(0, 0, labelNo);
								for(int i = 0; i < pixels.length; i++) {
									for(int j = 0; j < channels; j++) {
										row.set(0, j*pixels.length+i, ((pixels[i]>>>(8*j)) & 0xFF));
									}
								}
								if(images == null) {
									images = row;
									labels = lRow;
								}
								else {
									labels = (DenseDoubleMatrix2D) f.appendRows(labels, lRow);
									images = (DenseDoubleMatrix2D) f.appendRows(images, row);
								}
							}
						}
					}
				}
			}
		}
	}
	
	public DenseDoubleMatrix2D sample(int patchSize, int numPatches) {
		Random rand = new Random();
		DoubleMatrix2D patches = null;
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		DoubleFactory2D f = DoubleFactory2D.dense;
		for(int i = 0; i < numPatches; i++) {
			if(i%1000==0)
				System.out.println(i);
			int randomImage = rand.nextInt(images.rows());
			int randomY = rand.nextInt(height-patchSize+1);
			int randomX = rand.nextInt(width-patchSize+1);
			int imageSize = width*height;
			DoubleMatrix2D patch = null;
			for(int j = 0; j < channels; j++) {
				DoubleMatrix2D channel = d.subMatrix(images,randomImage, randomImage+1, j*imageSize, (j+1)*imageSize);
				DoubleMatrix1D c1 = channel.vectorize();
				channel = c1.reshape(height, width);
				channel = d.subMatrix(channel, randomY, randomY+patchSize, randomX, randomX+patchSize);
				c1 = channel.vectorize();
				channel = c1.reshape(1, patchSize*patchSize);
				if(patch == null) {
					patch = channel;
				}
				else {
					patch = f.appendColumns(patch, channel);
				}
			}
			if(patches == null) {
				patches = patch;
			}
			else {
				patches = f.appendRows(patches, patch);
			}
		}
		return (DenseDoubleMatrix2D) patches;
	}
	
	public DenseDoubleMatrix2D normalizeData(DenseDoubleMatrix2D data) {
		DenseDoubleMatrix2D mean = getRowMeans(data);
		data = (DenseDoubleMatrix2D) data.assign(mean, DeviceUtils.sub);
		DenseDoubleMatrix2D squareData = new DenseDoubleMatrix2D(data.rows(), data.columns());
		data.zMult(data, squareData);
		
		double var = squareData.zSum()/squareData.size();
		double stdev = Math.sqrt(var);
		double pstd = 3 * stdev;
		for(int i = 0; i < data.rows(); i++) {
			for(int j = 0; j < data.columns(); j++) {
				double x = data.get(i, j);
				double val = x<pstd?x:pstd;
				val = val>-pstd?val:-pstd;
				val /= pstd;
				data.set(i, j, val);
			}
		}
		data.assign(DoubleFunctions.plus(1));
		data.assign(DoubleFunctions.mult(0.4));
		data.assign(DoubleFunctions.plus(0.1));
		return data;
	}
	
	private DenseDoubleMatrix2D getRowMeans(DenseDoubleMatrix2D data) {
		DenseDoubleMatrix2D means = new DenseDoubleMatrix2D(data.rows(), data.columns());
		for(int i = 0; i < data.rows(); i++) {
			double mean = 0;
			for(int j = 0; j < data.columns(); j++) {
				mean += data.get(i, j)/data.columns();
			}
			for(int j = 0; j < data.columns(); j++) {
				means.set(i, j, mean);
			}
		}
		return means;
	}
	
	public DenseDoubleMatrix2D getImages() {
		return images;
	}
	
	public DenseDoubleMatrix2D getLabels() {
		DenseDoubleMatrix2D expandLabels = new DenseDoubleMatrix2D((int) labels.size(), (int) labels.getMaxLocation()[0]+1);
		for(int i = 0; i < labels.size(); i++) {
			expandLabels.set(i, (int)labels.get(i, 0), 1);
		}
		return expandLabels;
	}
	
	public HashMap<String, Double> getLabelMap() {
		return labelMap;
	}
}
