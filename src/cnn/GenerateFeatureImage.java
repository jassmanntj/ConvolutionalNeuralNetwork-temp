package cnn;

import java.io.IOException;

public class GenerateFeatureImage {
	public static void main(String[] args) throws IOException {
		LinearDecoder ae = new LinearDecoder(8, 3, 400, .035, 3e-3, 5, .5);
		ae.loadTheta("Layer0.csv", null);
		ae.visualize(8, 20, "Features.png");
	}

}
