package cnn.device;

import org.jtransforms.fft.DoubleFFT_2D;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplexFunctions;
import cern.jet.math.tdouble.DoubleFunctions;

public class DeviceUtils {
	
	public static DoubleDoubleFunction sub = new DoubleDoubleFunction() {
	    public double apply(double a, double b) { return a - b; }
	};   
	
	
	
	public static DenseDoubleMatrix2D sigmoid(DenseDoubleMatrix2D z) {
		z.assign(DoubleFunctions.neg);
		z.assign(DoubleFunctions.exp);
		z.assign(DoubleFunctions.plus(1));
		z.assign(DoubleFunctions.inv);
		return z;
	}
	
	
	public static DenseDoubleMatrix2D conv2d(DenseDoubleMatrix2D input, DenseDoubleMatrix2D kernel) {
		int totalRows = input.rows() + kernel.rows() - 1;
		int totalCols = input.columns() + kernel.columns() - 1;
		int rowSize = input.rows() - kernel.rows() + 1;
		int colSize = input.columns() - kernel.columns() + 1;
		int startRows = (totalRows-rowSize)/2;
		int startCols = (totalCols-colSize)/2;
		DoubleFactory2D concatFactory = DoubleFactory2D.dense;
		DenseDoubleMatrix2D flippedKernel = new DenseDoubleMatrix2D(kernel.rows(), kernel.columns());
		for(int i = 0; i < kernel.rows(); i++) {
			for(int j = 0; j < kernel.columns(); j++) {
				flippedKernel.set(i,j, kernel.get(kernel.rows()-1-i,kernel.columns()-1-j));
			}
		}
		kernel = flippedKernel;
		input = (DenseDoubleMatrix2D) concatFactory.appendColumns(input, new DenseDoubleMatrix2D(input.rows(), kernel.columns()-1));
		input = (DenseDoubleMatrix2D) concatFactory.appendRows(input, new DenseDoubleMatrix2D(kernel.rows()-1, input.columns()));
		kernel = (DenseDoubleMatrix2D) concatFactory.appendColumns(kernel, new DenseDoubleMatrix2D(kernel.rows(), input.columns()-kernel.columns()));
		kernel = (DenseDoubleMatrix2D) concatFactory.appendRows(kernel, new DenseDoubleMatrix2D(input.rows()-kernel.rows(), kernel.columns()));
		DenseDComplexMatrix2D inputDFT = new DenseDComplexMatrix2D(input);
		DenseDComplexMatrix2D kernelDFT = new DenseDComplexMatrix2D(kernel);
		DenseDComplexMatrix1D inDFT = (DenseDComplexMatrix1D) inputDFT.vectorize();
		DenseDComplexMatrix1D kernDFT = (DenseDComplexMatrix1D) kernelDFT.vectorize();
		
		DoubleFFT_2D t = new DoubleFFT_2D(input.rows(), input.columns());
		t.complexForward(inDFT.elements());
		t.complexForward(kernDFT.elements());
		
		inputDFT = (DenseDComplexMatrix2D) inDFT.reshape(inputDFT.rows(), inputDFT.columns());
		kernelDFT = (DenseDComplexMatrix2D) kernDFT.reshape(kernelDFT.rows(), kernelDFT.columns());
		kernelDFT.assign(inputDFT, DComplexFunctions.mult);
		kernDFT = (DenseDComplexMatrix1D) kernelDFT.vectorize();
		
		
		t.complexInverse(kernDFT.elements(), true);
		kernelDFT = (DenseDComplexMatrix2D) kernDFT.reshape(kernelDFT.rows(), kernelDFT.columns());
		
		DenseDoubleMatrix2D result = (DenseDoubleMatrix2D) kernelDFT.getRealPart();
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		return (DenseDoubleMatrix2D) d.subMatrix(result,startRows,startRows+rowSize-1,startCols,startCols+colSize-1);
	 }
	
	public static DenseDoubleMatrix2D conv2d2(DenseDoubleMatrix2D input, DenseDoubleMatrix2D kernel) {
		int rowSize = input.rows() - kernel.rows() + 1;
		int colSize = input.columns() - kernel.columns() + 1;
		int totalRows = input.rows() + kernel.rows() - 1;
		int totalCols = input.columns() + kernel.columns() - 1;
		int maxSize = Math.max(input.rows(), input.columns());
		int padRows = 1 << (int)Math.ceil(Math.log(maxSize)/Math.log(2));
		System.out.println(padRows);
		int startRows = (totalRows-rowSize)/2;
		int startCols = (totalCols-colSize)/2;
		DoubleFactory2D concatFactory = DoubleFactory2D.dense;
		DenseDoubleMatrix2D flippedKernel = new DenseDoubleMatrix2D(kernel.rows(), kernel.columns());
		
		for(int i = 0; i < kernel.rows(); i++) {
			for(int j = 0; j < kernel.columns(); j++) {
				flippedKernel.set(i,j, kernel.get(kernel.rows()-1-i,kernel.columns()-1-j));
			}
		}
		kernel = flippedKernel;
		input = (DenseDoubleMatrix2D) concatFactory.appendColumns(input, new DenseDoubleMatrix2D(input.rows(), padRows-input.columns()));
		input = (DenseDoubleMatrix2D) concatFactory.appendRows(input, new DenseDoubleMatrix2D(padRows-input.rows(), input.columns()));
		kernel = (DenseDoubleMatrix2D) concatFactory.appendColumns(kernel, new DenseDoubleMatrix2D(kernel.rows(), padRows-kernel.columns()));
		kernel = (DenseDoubleMatrix2D) concatFactory.appendRows(kernel, new DenseDoubleMatrix2D(padRows-kernel.rows(), kernel.columns()));
		
		DenseDComplexMatrix2D inputDFT = input.getFft2();
		DenseDComplexMatrix2D kernelDFT = kernel.getFft2();
		kernelDFT.assign(inputDFT, DComplexFunctions.mult);
		kernelDFT.ifft2(true);
		kernel = (DenseDoubleMatrix2D) kernelDFT.getRealPart();
		DenseDoubleAlgebra d = new DenseDoubleAlgebra();
		return (DenseDoubleMatrix2D) d.subMatrix(kernel,startRows,startRows+rowSize-1,startCols,startCols+colSize-1);
	}
	
	public static int[] computeResults(DoubleMatrix2D result) {
		int[] results = new int[result.rows()];
		for(int i = 0; i < result.rows(); i++) {
			double currentMax = 0;
			for(int j = 0; j < result.columns(); j++) {
				if(result.get(i,j) > currentMax) {
					currentMax = result.get(i,j);
					results[i] = j;
				}
			}
		}
		return results;
	}


	public static DenseDoubleMatrix2D ZCAWhiten(DenseDoubleMatrix2D input, DoubleMatrix1D meanPatch, DenseDoubleMatrix2D ZCAWhite) {
		for(int i = 0; i < input.rows(); i++) {
			input.viewRow(i).assign(meanPatch, DeviceUtils.sub);
		}
		DenseDoubleMatrix2D result = new DenseDoubleMatrix2D(input.rows(), ZCAWhite.columns());
		input.zMult(ZCAWhite, result);
		return result;
	}

}
