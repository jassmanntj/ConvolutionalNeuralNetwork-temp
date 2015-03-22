package cnn;

import cnn.Utils;
import org.jblas.DoubleMatrix;

/**
 * Created by jassmanntj on 3/22/2015.
 */
public class DataContainer {
    private DoubleMatrix[][] dataArray;
    private DoubleMatrix data;
    private int rows;
    private int columns;
    private int length;
    private int channels;

    public DataContainer(DoubleMatrix data) {
        this.data = data;
        this.length = data.rows;
    }

    public DataContainer(DoubleMatrix[][] data) {
        this.dataArray = data;
        this.length = data.length;
        this.rows = data[0][0].rows;
        this.columns = data[0][0].columns;
        this.channels = data[0].length;
    }

    public void update(DoubleMatrix data) {
        for(int i = 0; i < dataArray.length; i++) {
            for(int j = 0; j < dataArray[i].length; j++) {
                for(int k = 0; k < dataArray[i][j].rows; k++) {
                    for(int l = 0; l < dataArray[i][j].columns; l++) {
                        dataArray[i][j].put(k,l, data.get(i,j*dataArray[i][j].length+k*dataArray[i][j].rows+l));
                    }
                }
            }
        }
    }

    public int length() {
        return length;
    }

    public int channels() {
        return channels;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public DoubleMatrix getData() {
        if(data != null) return data;
        else return Utils.flatten(dataArray);
    }

    public DoubleMatrix[][] getDataArray() {
        if(dataArray != null) return dataArray;
        else {
            DoubleMatrix[][] res = new DoubleMatrix[data.rows][1];
            for(int i = 0; i < data.rows; i++) {
                res[i][0] = data.getRow(i);
            }
            return res;
        }
    }

}
