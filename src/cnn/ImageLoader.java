package cnn;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;
import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;

public class ImageLoader {
	DoubleMatrix images[];
	DoubleMatrix labels[];
    DoubleMatrix imgArr[][];
    DoubleMatrix imgs;
    DoubleMatrix lbls;
    DoubleMatrix testImgs;
    DoubleMatrix testLbls;
	int channels;
	int width, height;
    int[] counts;
	HashMap<String, Double> labelMap;
	ArrayList<String> names;


    public void loadFolder(File folder, int channels, int width, int height, HashMap<String, Double> labelMap) throws IOException {
        int z = 0;
        this.channels = channels;
        this.width = width;
        this.height = height;
        this.labelMap = labelMap;
        this.names = new ArrayList<String>();
        this.counts = countImages(folder, labelMap);
        int total = 0;
        for(int c : counts) {
            total += c;
        }
        this.imgArr = new DoubleMatrix[total][3];
        images = new DoubleMatrix[labelMap.size()];
        labels = new DoubleMatrix[labelMap.size()];
        for(int i = 0; i < labelMap.size(); i++) {
            images[i] = new DoubleMatrix(counts[i], channels * width * height);
            labels[i] = new DoubleMatrix(counts[i], labelMap.size());
        }
        DoubleMatrix imageNo = new DoubleMatrix(counts[0]);
        for(int i = 0; i < imageNo.length; i++) {
            imageNo.put(i, i);
        }

        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(File leaf : folder.listFiles()) {
            int i = 0;
            if(leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for(File image : leaf.listFiles()) {
                    Runnable ip = new ImageProcessor2(image, i, labelMap.get(leaf.getName()), imageNo, z++);
                    executor.execute(ip);
                    i++;
                }
            }
        }
        executor.shutdown();
        while(!executor.isTerminated());
    }

    private class ImageProcessor implements Runnable {
        private File image;
        private double leafNo;
        private int num;
        private int imgNo;
        private int z;
        public ImageProcessor(File image, int num, double leafNo, DoubleMatrix imageNo, int z) {
            this.image = image;
            this.num = num;
            this.imgNo = num;
            this.z = z;
            this.leafNo = leafNo;
        }

        public void run(){
            System.out.println(z);
            BufferedImage img = null;
            try {
                img = ImageIO.read(image);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (img.getHeight() < img.getWidth()) {
                img = Scalr.rotate(img, Rotation.CW_90);
            }
            if (img.getHeight() * 3 < img.getWidth() * 4) {
                BufferedImage newImage = new BufferedImage(img.getWidth(), img.getWidth() * 4 / 3, img.getType());
                Graphics g = newImage.getGraphics();
                g.setColor(Color.black);
                g.fillRect(0, 0, newImage.getWidth(), newImage.getHeight());
                g.drawImage(img, 0, (newImage.getHeight() - img.getHeight()) / 2, null);
                g.dispose();
                img = newImage;
                newImage.flush();
            } else if (img.getHeight() * 3 > img.getWidth() * 4) {
                BufferedImage newImage = new BufferedImage(img.getHeight() * 3 / 4, img.getHeight(), img.getType());
                Graphics g = newImage.getGraphics();
                g.setColor(Color.black);
                g.fillRect(0, 0, newImage.getWidth(), newImage.getHeight());
                g.drawImage(img, (newImage.getWidth() - img.getWidth()) / 2, 0, null);
                g.dispose();
                img = newImage;
                newImage.flush();
            }
            img = Scalr.resize(img, Method.QUALITY, width, height);
            int[] pixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
            img.flush();
            if (pixels.length == width * height) {
                for (int i = 0; i < pixels.length; i++) {
                    for (int j = 0; j < channels; j++) {
                        images[(int)leafNo].put(num, j*width*height+i, ((pixels[i] >>> (8 * j)) & 0xFF));
                    }
                }
                labels[(int)leafNo].put(num, (int)leafNo, 1);
            }
        }
    }

    private class ImageProcessor2 implements Runnable {
        private File image;
        private double leafNo;
        private int num;
        private int imgNo;
        private int z;
        public ImageProcessor2(File image, int num, double leafNo, DoubleMatrix imageNo, int z) {
            this.image = image;
            this.num = num;
            this.imgNo = num;
            this.z = z;
            this.leafNo = leafNo;
        }

        public void run(){
            System.out.println(z);
            BufferedImage img = null;
            try {
                img = ImageIO.read(image);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (img.getHeight() < img.getWidth()) {
                img = Scalr.rotate(img, Rotation.CW_90);
            }
            if (img.getHeight() * 3 < img.getWidth() * 4) {
                BufferedImage newImage = new BufferedImage(img.getWidth(), img.getWidth() * 4 / 3, img.getType());
                Graphics g = newImage.getGraphics();
                g.setColor(Color.black);
                g.fillRect(0, 0, newImage.getWidth(), newImage.getHeight());
                g.drawImage(img, 0, (newImage.getHeight() - img.getHeight()) / 2, null);
                g.dispose();
                img = newImage;
                newImage.flush();
            } else if (img.getHeight() * 3 > img.getWidth() * 4) {
                BufferedImage newImage = new BufferedImage(img.getHeight() * 3 / 4, img.getHeight(), img.getType());
                Graphics g = newImage.getGraphics();
                g.setColor(Color.black);
                g.fillRect(0, 0, newImage.getWidth(), newImage.getHeight());
                g.drawImage(img, (newImage.getWidth() - img.getWidth()) / 2, 0, null);
                g.dispose();
                img = newImage;
                newImage.flush();
            }
            img = Scalr.resize(img, Method.QUALITY, width, height);
            int[] pixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
            img.flush();
            for(int i = 0; i < channels; i++) {
                imgArr[z][i] = new DoubleMatrix(height, width);
            }
            if (pixels.length == width * height) {
                for (int i = 0; i < pixels.length; i++) {
                    for (int j = 0; j < channels; j++) {
                        imgArr[z][j].put(i/width, i%width, ((pixels[i] >>> (8 * j)) & 0xFF));
                    }
                }
                labels[(int)leafNo].put(num, (int)leafNo, 1);
            }
        }
    }

    public DoubleMatrix[][] getImgArr() {
        return imgArr;
    }

    private int[] countImages(File folder, HashMap<String, Double> labelMap) {
        int count[] = new int[labelMap.size()];
        for(File leaf : folder.listFiles()) {
            if (leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for (File image : leaf.listFiles()) {
                    count[labelMap.get(leaf.getName()).intValue()]++;
                }
            }
        }
        return count;
    }

	public ArrayList<String> getNames(File folder) {
		if(names == null) {
			names = new ArrayList<String>();
			for(File leaf : folder.listFiles()) {
				if(leaf.isDirectory()) {
                    for(File image : leaf.listFiles()) {
                        names.add(image.getName());
                    }
				}
			}
		}
		return names;
	}

    public static DoubleMatrix sample(final int patchRows, final int patchCols, final int numPatches, final DoubleMatrix images, final int width, final int height, final int channels) {
        final DoubleMatrix patches = new DoubleMatrix(numPatches, channels*patchRows*patchCols);
        class Patcher implements Runnable {
            private int threadNo;

            public Patcher(int threadNo) {
                this.threadNo = threadNo;
            }

            @Override
            public void run() {
                int count = numPatches / Utils.NUMTHREADS;
                for (int i = 0; i < count; i++) {
                    if(i%500 == 0)
                        System.out.println(threadNo+":"+i);
                    Random rand = new Random();
                    int randomImage = rand.nextInt(images.rows);
                    int randomY = rand.nextInt(height - patchRows + 1);
                    int randomX = rand.nextInt(width - patchCols + 1);
                    DoubleMatrix ch = null;
                    for (int j = 0; j < channels; j++) {
                        DoubleMatrix channel = images.getRange(randomImage,randomImage+1,j*height*width,(j+1)*height*width).reshape(height, width);
                        channel = channel.getRange(randomY, randomY + patchRows, randomX, randomX + patchCols);
                        channel = channel.reshape(1, patchRows * patchCols);
                        if(ch == null) ch = channel;
                        else ch = DoubleMatrix.concatHorizontally(ch, channel);
                    }
                    patches.putRow(threadNo*count+i, ch);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int i = 0; i < Utils.NUMTHREADS; i++) {
            if (i % 1000 == 0)
                System.out.println(i);
            Runnable patcher = new Patcher(i);
            executor.execute(patcher);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return normalizeData(patches);
    }


	public static DoubleMatrix normalizeData(DoubleMatrix data) {
		DoubleMatrix mean = data.rowMeans();
		data.subiColumnVector(mean);
		DoubleMatrix squareData = data.mul(data);

		double var = squareData.mean();
		double stdev = Math.sqrt(var);
		double pstd = 3 * stdev;
		for(int i = 0; i < data.rows; i++) {
			for(int j = 0; j < data.columns; j++) {
				double x = data.get(i, j);
				double val = x<pstd?x:pstd;
				val = val>-pstd?val:-pstd;
				val /= pstd;
				data.put(i, j, val);
			}
		}
		data.addi(1).muli(.4).add(0.1);
		return data;
	}

	public DoubleMatrix getImages() {
        DoubleMatrix imgs = images[0];
        for(int i = 1; i < labelMap.size(); i++) {
            imgs = DoubleMatrix.concatVertically(imgs, images[i]);
        }
		return normalizeData(imgs);
	}

    public void sortImgs(int i) {
        testImgs = null;
        testLbls = null;
        imgs = null;
        lbls = null;
        DoubleMatrix testNo = new DoubleMatrix(counts.length);
        for(int j = 0; j < testNo.length ;j++) {
            testNo.put(j, j);
        }
        Random rand = new Random(System.currentTimeMillis());
        for(int j = 0; j < 10000; j++) {
            int a = rand.nextInt(testNo.length);
            int b = rand.nextInt(testNo.length);
            testNo.swapRows(a, b);
        }
        for(double j : testNo.data) {
            DoubleMatrix imageNo = new DoubleMatrix(counts[(int)j]);
            for(int k = 0; k < imageNo.length; k++) {
                imageNo.put(k, k);
            }
            for(int k = 0; k < 10000; k++) {
                int a = rand.nextInt(imageNo.length);
                int b = rand.nextInt(imageNo.length);
                imageNo.swapRows(a, b);
                imageNo.swapRows(a, b);
            }
            for(double k : imageNo.data) {
                if (k == i || k == i+1 || k == i+2) {
                    if(testImgs == null) {
                        testImgs = images[(int)j].getRow((int)k);
                        testLbls = labels[(int)j].getRow((int)k);
                    }
                    else {
                        testImgs = DoubleMatrix.concatVertically(testImgs, images[(int)j].getRow((int)k));
                        testLbls = DoubleMatrix.concatVertically(testLbls, labels[(int)j].getRow((int)k));
                    }
                }
                else{
                    if(imgs == null) {
                        imgs = images[(int)j].getRow((int)k);
                        lbls = labels[(int)j].getRow((int)k);
                    }
                    else {
                        imgs = DoubleMatrix.concatVertically(imgs, images[(int)j].getRow((int)k));
                        lbls = DoubleMatrix.concatVertically(lbls, labels[(int)j].getRow((int)k));
                    }
                }
            }
        }
    }

    public DoubleMatrix getTestImgs() {
        return normalizeData(testImgs);
    }
    public DoubleMatrix getImgs() {
        return normalizeData(imgs);
    }
    public DoubleMatrix getLbls() {
        return lbls;
    }
    public DoubleMatrix getTestLbls() {
        return testLbls;
    }

	public DoubleMatrix getLabels() {
        DoubleMatrix lbls = labels[0];
        for(int i = 1; i < labelMap.size(); i++) {
            lbls = DoubleMatrix.concatVertically(lbls, labels[i]);
        }
		return lbls;
	}

	public HashMap<String, Double> getLabelMap(File folder) {
		HashMap<String, Double> labelMap = new HashMap<String, Double>();
		double labelNo = -1;
		for(File leaf : folder.listFiles()) {
			labelNo++;
			leaf.listFiles();
			labelMap.put(leaf.getName(), labelNo);
		}
		return labelMap;
	}
}
