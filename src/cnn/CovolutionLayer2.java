package cnn;

import org.jblas.DoubleMatrix;

import java.io.*;

/**
 * Created by jassmanntj on 3/25/2015.
 */
public class CovolutionLayer2 extends NeuralNetworkConvLayer {
    private DoubleMatrix theta[][];
    private DoubleMatrix thetaVelocity[][];
    private double bias[][];
    private double biasVelocity[][];
    private int channels;
    private int numFeatures;

    public CovolutionLayer2(int numFeatures, int channels, int patchDim) {
        this.numFeatures = numFeatures;
        this.channels = channels;

        bias = new double[numFeatures][channels];
        biasVelocity = new double[numFeatures][channels];
        theta = new DoubleMatrix[numFeatures][channels];
        thetaVelocity = new DoubleMatrix[numFeatures][channels];

        for(int i = 0; i < numFeatures; i++) {
            for(int j = 0; j < channels; j++) {
                theta[i][j] = initializeTheta(patchDim);
            }
        }

    }

    private DoubleMatrix initializeTheta(int patchDim) {
        return DoubleMatrix.rand(patchDim, patchDim);
    }

    @Override
    public DoubleMatrix[][] compute(DoubleMatrix[][] input) {
        System.out.println("Starting Convolution");
        DoubleMatrix[][] result = new DoubleMatrix[input.length][theta.length];
        for(int image = 0; image < input.length; image++) {
            for(int feature = 0; feature < numFeatures; feature++) {
                for(int channel = 0; channel < channels; channel++) {
                    result[image][feature].addi(Utils.conv2d(input[image][channel], theta[feature][channel], true));
                }
                result[image][feature] = Utils.sigmoid(result[image][feature].add(bias[image][feature]));
            }
        }
        return result;
    }

    @Override
    public DoubleMatrix[][] train(DoubleMatrix[][] input, int iterations) throws IOException {
        return null;
    }

    @Override
    public DoubleMatrix[][] backPropagation(DoubleMatrix[][] input, DoubleMatrix delta[][], double momentum, double alpha) {
        DoubleMatrix[][] delt = new DoubleMatrix[input.length][channels];
        for(int image = 0; image < input.length; image++) {
            for(int feature = 0; feature < numFeatures; feature++) {
                for(int channel = 0; channel < channels; channel++) {
                    delt[image][channel].addi(Utils.conv2d(delta[image][feature], theta[feature][channel], false));
                }
            }
        }

        for(int image = 0; image < input.length; image++) {
            for(int channel = 0; channel < channels; channel++) {
                delt[image][channel].muli(Utils.sigmoidGradient(input[image][channel]));
            }
        }

        DoubleMatrix[][] thetaGrad = new DoubleMatrix[input.length][channels];
        for(int image = 0; image < input.length; image++) {
            for(int feature = 0; feature < numFeatures; feature++) {
                for(int channel = 0; channel < channels; channel++) {
                    thetaGrad[feature][channel].addi(Utils.conv2d(input[image][channel], Utils.reverseMatrix(delta[image][feature]), true));
                }
            }
        }

        for(int i = 0; i < numFeatures; i++) {
            for(int j = 0; j < channels; j++) {
                biasVelocity[i][j] *= momentum;
                biasVelocity[i][j] += delt[i][j].mean() * alpha/input.length;
                thetaVelocity[i][j].muli(momentum).addi(thetaGrad[i][j].mul(alpha/input.length));
                theta[i][j].subi(thetaVelocity[i][j]);
                bias[i][j] -= biasVelocity[i][j];
            }
        }
        return delt;
    }

    @Override
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

    @Override
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
