package cnn;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class CreateLabelMap {
	public static void main(String[] args) throws IOException {
		ImageLoader loader = new ImageLoader();
		File folder = new File("C:/Users/jassmanntj/Desktop/TrainSort");
		HashMap<String, Double> labelMap = loader.getLabelMap(folder);
		writeMap(labelMap);
	}

	private static void writeMap(HashMap<String, Double> labelMap) throws IOException {
		FileWriter fw = new FileWriter("LabelMap");
		BufferedWriter writer = new BufferedWriter(fw);
		for(String s : labelMap.keySet()) {
			writer.write(s+":"+labelMap.get(s)+"\n");
		}
		writer.close();
		
	}
	
}
