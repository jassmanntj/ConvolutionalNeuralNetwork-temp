package cnn;

import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by jassmanntj on 3/25/2015.
 */
public class ConvolutionLayer extends ConvPoolLayer {
    private DoubleMatrix theta[][];
    private DoubleMatrix thetaVelocity[][];
    private double bias[];
    private double biasVelocity[];
    private int patchDim;

    public ConvolutionLayer(int numFeatures, int channels, int patchDim) {
        this.patchDim = patchDim;

        bias = new double[numFeatures];
        biasVelocity = new double[numFeatures];
        theta = new DoubleMatrix[numFeatures][channels];
        thetaVelocity = new DoubleMatrix[numFeatures][channels];

        for(int i = 0; i < numFeatures; i++) {
            for(int j = 0; j < channels; j++) {
                theta[i][j] = initializeTheta(patchDim);
                thetaVelocity[i][j] = new DoubleMatrix(patchDim, patchDim);
            }
        }

    }

    private DoubleMatrix initializeTheta(int patchDim) {
        return DoubleMatrix.rand(patchDim, patchDim);
    }

    public DoubleMatrix[][] compute(DoubleMatrix[][] input) {
        DoubleMatrix[][] result = new DoubleMatrix[input.length][theta.length];

        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < theta.length; feature++) {
                    DoubleMatrix res = new DoubleMatrix(input[imageNum][0].rows-theta[0][0].rows+1, input[imageNum][0].columns-theta[0][0].columns+1);
                    for(int channel = 0; channel < theta[feature].length; channel++) {
                        res.addi(Utils.conv2d(input[imageNum][channel], theta[feature][channel], true));
                    }
                    result[imageNum][feature] = Utils.sigmoid(res.add(bias[feature]));
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return result;
    }


    public DoubleMatrix[][] gradientCheck(DoubleMatrix[][] in, DoubleMatrix labels, DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix[][] delta, NeuralNetwork cnn) {
        DoubleMatrix[][] delt = new DoubleMatrix[input.length][theta[0].length];
        DoubleMatrix[][] thetaGrad = new DoubleMatrix[theta.length][theta[0].length];
        for(int image = 0; image < input.length; image++) {
            for (int channel = 0; channel < theta[0].length; channel++) {
                delt[image][channel] = new DoubleMatrix(input[0][0].rows, input[0][0].columns);
            }
        }
        for(int feature = 0; feature < theta.length; feature++) {
            for (int channel = 0; channel < theta[0].length; channel++) {
                thetaGrad[feature][channel] = new DoubleMatrix(patchDim, patchDim);
            }
        }
        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < theta.length; feature++) {
                    delta[imageNum][feature].muli(Utils.sigmoidGradient(output[imageNum][feature]));
                    for(int channel = 0; channel < theta[feature].length; channel++) {
                        delt[imageNum][channel].addi(Utils.conv2d(delta[imageNum][feature], Utils.reverseMatrix(theta[feature][channel]), false));
                        thetaGrad[feature][channel].addi(Utils.conv2d(input[imageNum][channel], delta[imageNum][feature], true));
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());


        double epsilon = 0.0001;
        DoubleMatrix biasG = new DoubleMatrix(1, bias.length);
        DoubleMatrix biasGrad = new DoubleMatrix(1, bias.length);
        for(int i = 0; i < theta.length; i++) {
            for (int j = 0; j < input.length; j++) {
                biasGrad.put(i, biasGrad.get(i) + delta[j][i].sum()/input.length);
            }
        }
        for(int i = 0; i < bias.length; i++) {
            bias[i]+=epsilon;
            CostResult costPlus = cnn.computeCost(in, labels);
            bias[i]-=2*epsilon;
            CostResult costMinus = cnn.computeCost(in, labels);
            bias[i]+=epsilon;
            biasG.put(i, (costPlus.cost-costMinus.cost)/(2*epsilon));
        }
        DoubleMatrix biasA = biasG.add(biasGrad);
        DoubleMatrix biasS = biasG.sub(biasGrad);
        System.out.println("CL Bias Diff: "+biasS.norm2()/biasA.norm2());

        for(int i = 0; i < theta.length; i++) {
            for(int j = 0; j < theta[i].length; j++) {
                DoubleMatrix thetaG = new DoubleMatrix(thetaGrad[i][j].rows, thetaGrad[i][j].columns);
                for(int k = 0; k < theta[i][j].length; k++) {
                    theta[i][j].put(k, theta[i][j].get(k)+epsilon);
                    CostResult costPlus = cnn.computeCost(in, labels);
                    theta[i][j].put(k, theta[i][j].get(k)-2*epsilon);
                    CostResult costMinus = cnn.computeCost(in, labels);
                    theta[i][j].put(k, theta[i][j].get(k)+epsilon);
                    thetaG.put(k, (costPlus.cost-costMinus.cost)/(2*epsilon));
                }
                DoubleMatrix thetaA = thetaG.add(thetaGrad[i][j].div(input.length));
                DoubleMatrix thetaS = thetaG.sub(thetaGrad[i][j].div(input.length));
                System.out.println("CL Theta "+i+"/"+theta.length+":"+j+"/"+theta[i].length+" Diff: "+thetaS.norm2()/thetaA.norm2());
            }
        }
        return delt;
    }

    public DoubleMatrix[][] backPropagation(DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix delta[][], double momentum, double alpha) {
        DoubleMatrix[][] delt = new DoubleMatrix[input.length][theta[0].length];
        DoubleMatrix[][] thetaGrad = new DoubleMatrix[theta.length][theta[0].length];
        for(int image = 0; image < input.length; image++) {
            for (int channel = 0; channel < theta[0].length; channel++) {
                delt[image][channel] = new DoubleMatrix(input[0][0].rows, input[0][0].columns);
            }
        }
        for(int feature = 0; feature < theta.length; feature++) {
            for (int channel = 0; channel < theta[0].length; channel++) {
                thetaGrad[feature][channel] = new DoubleMatrix(patchDim, patchDim);
            }
        }
        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < theta.length; feature++) {
                    delta[imageNum][feature].muli(Utils.sigmoidGradient(output[imageNum][feature]));
                    for(int channel = 0; channel < theta[feature].length; channel++) {
                        delt[imageNum][channel].addi(Utils.conv2d(delta[imageNum][feature], Utils.reverseMatrix(theta[feature][channel]), false));
                        thetaGrad[feature][channel].addi(Utils.conv2d(input[imageNum][channel], delta[imageNum][feature], true));
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());



        for(int i = 0; i < theta.length; i++) {
            biasVelocity[i] *= momentum;
            double deltMean = 0;
            for(int j = 0; j < input.length; j++) {
                deltMean += delta[j][i].sum();
            }
            biasVelocity[i] += deltMean * alpha/input.length;
            bias[i] -= biasVelocity[i];
            for(int j = 0; j < theta[i].length; j++) {
                thetaVelocity[i][j].muli(momentum).addi(thetaGrad[i][j].mul(alpha/input.length));
                theta[i][j].subi(thetaVelocity[i][j]);
            }
        }
        return delt;
    }

    public void writeLayer(String filename) {
        /*try {
            FileWriter fw = new FileWriter(filename);
            BufferedWriter writer = new BufferedWriter(fw);
            writer.write(theta.length+","+theta.columns+"\n");
            for(int i = 0; i < theta.rows; i++) {
                for(int j = 0; j < theta.columns; j++) {
                    writer.write(theta.get(i,j)+",");
                }
            }
            writer.write("\n"+bias.rows+","+bias.columns+"\n");
            for(int i = 0; i < bias.rows; i++) {
                for(int j = 0; j < bias.columns; j++) {
                    writer.write(bias.get(i,j)+",");
                }
            }
            writer.write("\n"+imageRows+"\n"+imageCols+"\n"+patchDim+"\n"+poolDim);
            writer.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }*/
    }

    public void loadLayer(String filename) {
        /*try {
            FileReader fr = new FileReader(filename);
            @SuppressWarnings("resource")
            BufferedReader reader = new BufferedReader(fr);
            String[] line = reader.readLine().split(",");
            theta = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
            line = reader.readLine().split(",");
            for(int i = 0; i < theta.rows; i++) {
                for(int j = 0; j < theta.columns; j++) {
                    theta.put(i, j, Double.parseDouble(line[i * theta.columns + j]));
                }
            }
            line = reader.readLine().split(",");
            bias = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
            line = reader.readLine().split(",");
            for(int i = 0; i < bias.rows; i++) {
                for(int j = 0; j < bias.columns; j++) {
                    bias.put(i, j, Double.parseDouble(line[i * bias.columns + j]));
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
        }*/
    }
}
