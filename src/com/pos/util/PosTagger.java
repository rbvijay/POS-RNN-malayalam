package com.pos.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.apache.commons.lang.StringUtils;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.examples.neural.recurrent.elman.ElmanXOR;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.Greedy;
import org.encog.ml.train.strategy.HybridStrategy;
import org.encog.ml.train.strategy.StopTrainingStrategy;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.neural.pattern.JordanPattern;
import org.encog.persist.EncogDirectoryPersistence;

public class PosTagger {

	static List<String> inputCorpus = new ArrayList<>();
	static List<String> posCorpus = new ArrayList<>();
	static final String START = "START";
	// No. of words from training corpus
	static int trainSize = 10041;
	// No. of unique words from training corpus + unique POS tags
	static int NN_Input_Size = 4311;
	static int NN_Hidden_Size = 100;
	// No. of unique POS tags
	static int NN_Output_Size = 30;
	static double NN_INPUT[][] = new double[trainSize][NN_Input_Size];
	static double NN_IDEAL[][] = new double[trainSize][NN_Output_Size];
	static String nn_seralized;
	static String nn_type;

	public PosTagger(String trainNw) {
		super();
		nn_type = trainNw;
		switch (trainNw) {
		case "elman":
			nn_seralized = "resources/elman.eg";
			break;
		case "jordan":
			nn_seralized = "resources/jordan.eg";
			break;
		default: nn_seralized = "resources/resilent.eg";
			break;
		}
		// formCorpus();
		loadCorpus();
		loadEmptyData();
	}

	private void loadEmptyData() {
		for (int i = 0; i < trainSize; i++) {
			for (int j = 0; j < NN_Input_Size; j++) {
				NN_INPUT[i][j] = 0.0;
			}
		}
		for (int i = 0; i < trainSize; i++) {
			for (int j = 0; j < NN_Output_Size; j++) {
				NN_IDEAL[i][j] = 0.0;
			}
		}
	}

	static BasicNetwork createElmanNetwork() {
		// Elman type network
		ElmanPattern pattern = new ElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(NN_Input_Size);
		pattern.addHiddenLayer(NN_Hidden_Size);
		pattern.setOutputNeurons(NN_Output_Size);
		return (BasicNetwork) pattern.generate();
	}

	static BasicNetwork createJordanNetwork() {
		// Jordan type network
		JordanPattern pattern = new JordanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(NN_Input_Size);
		pattern.addHiddenLayer(NN_Hidden_Size);
		pattern.setOutputNeurons(NN_Output_Size);
		return (BasicNetwork) pattern.generate();
	}

	public void trainNN() {
		try {
			String sentenceEnd = ".";
			String prevPOS = START;
			File inputfile = new File("resources/mal_pos.train");
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputfile), "UTF8"));
			String str;
			int lineCount = 0;
			while ((str = br.readLine()) != null) {
				str = str.trim();
				if ("".equals(str))
					continue;
				StringTokenizer stt = new StringTokenizer(str);
				String word = stt.nextToken();
				String pos = stt.nextToken();
				int wordNum = inputCorpus.indexOf(word);
				int prevPOSNum = inputCorpus.indexOf(prevPOS);
				NN_INPUT[lineCount][wordNum] = 1.0;
				NN_INPUT[lineCount][prevPOSNum] = 1.0;

				int posNum = posCorpus.indexOf(pos);
				NN_IDEAL[lineCount][posNum] = 1.0;
				if (word.equals(sentenceEnd)) {
					prevPOS = START;
				} else {
					prevPOS = pos;
				}
				lineCount++;
			}
			br.close();

			NeuralDataSet trainingSet = new BasicNeuralDataSet(NN_INPUT,
					NN_IDEAL);
			switch (nn_type) {
			case "elman":
				this.trainElmanNN(trainingSet);
				break;
			case "jordan":
				this.trainJordanNN(trainingSet);
				break;
			default: this.trainResilentNN(trainingSet);
				break;
			}


		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void trainElmanNN(NeuralDataSet trainingSet) {
		final BasicNetwork elmanNetwork = PosTagger.createElmanNetwork();
		ResilientPropagation prop=new ResilientPropagation(elmanNetwork,trainingSet);
		int epoch = 1;
		do {
			prop.iteration();
			System.out
					.println("Epoch #" + epoch + " Error:" + prop.getError());
			epoch++;
		} while (epoch < 12);
		prop.finishTraining();		
		File nn_backup = new File(nn_seralized);
		nn_backup.delete();
		EncogDirectoryPersistence.saveObject(nn_backup, elmanNetwork);
		Encog.getInstance().shutdown();
	}

	public void trainJordanNN(NeuralDataSet trainingSet) {
		final BasicNetwork jordanNetwork = PosTagger.createJordanNetwork();
		ResilientPropagation prop=new ResilientPropagation(jordanNetwork,trainingSet);
		int epoch = 1;
		do {
			prop.iteration();
			System.out
					.println("Epoch #" + epoch + " Error:" + prop.getError());
			epoch++;
		} while (epoch < 20);
		prop.finishTraining();
		File nn_backup = new File(nn_seralized);
		nn_backup.delete();
		EncogDirectoryPersistence.saveObject(nn_backup, jordanNetwork);
		Encog.getInstance().shutdown();
	}

	public void trainResilentNN(NeuralDataSet trainingSet) {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 4311));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 100));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 30));
		network.getStructure().finalizeStructure();
		network.reset();

		// train the neural network
		final ResilientPropagation train = new ResilientPropagation(network,
				trainingSet);

		int epoch = 1;

		do {
			train.iteration();
			System.out
					.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		} while (train.getError() > 0.01);
		train.finishTraining();
		File nn_backup = new File(nn_seralized);
		nn_backup.delete();
		EncogDirectoryPersistence.saveObject(nn_backup, network);
		Encog.getInstance().shutdown();
	}

	public void testNN() {
		try {
			int testSize = 100000;
			boolean unseenWord = false;
			String sentenceEnd = ".";
			String prevPOS = START;
			BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence
					.loadObject(new File(nn_seralized));
			
			String optext = "resources/op.txt";
			File opFile = new File(optext);
			opFile.delete();
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(opFile, true), "UTF-8"));
			
			File inputfile = new File("resources/mal_pos.gold");
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputfile), "UTF8"));
			String str;
			int lineCount = 0;
			int correctCount = 0;
			while ((str = br.readLine()) != null && lineCount < testSize) {
				str = str.trim();
				double NN_INPUT[][] = new double[1][NN_Input_Size];
				double NN_IDEAL[][] = new double[1][NN_Output_Size];
				if ("".equals(str))
					continue;
				StringTokenizer stt = new StringTokenizer(str);
				String word = stt.nextToken();
				String pos = stt.nextToken();
				int wordNum = inputCorpus.indexOf(word);
				int prevPOSNum = inputCorpus.indexOf(prevPOS);
				if (wordNum > 0) {
					NN_INPUT[0][wordNum] = 1.0;
					NN_INPUT[0][prevPOSNum] = 1.0;
					unseenWord = false;
				} else {
					unseenWord = true;
//					String similarWord = getClosestWord(word);
					String similarWord = getClosestEnd(word);
					if ("".equals(similarWord)) 
						similarWord = getClosestWord(word);
					wordNum = inputCorpus.indexOf(similarWord);
					prevPOSNum = inputCorpus.indexOf(prevPOS);
					NN_INPUT[0][wordNum] = 1.0;
					NN_INPUT[0][prevPOSNum] = 1.0;
				}
				pos = pos.toUpperCase();
				int posNum = posCorpus.indexOf(pos);
				NN_IDEAL[0][posNum] = 1.0;
				NeuralDataSet trainingSet = new BasicNeuralDataSet(NN_INPUT,
						NN_IDEAL);
				MLDataPair pair = trainingSet.get(0);
				final MLData output = network.compute(pair.getInput());
				String consoleOP = "Word=" + word + ",		predicted="
						+ posCorpus.get(getPosIndex(output.getData()))
						+ ",	actual=" + pos + ",		unseenWord=" + unseenWord;
/*				String consoleOP = "predicted="
						+ getPosIndex(output.getData())
						+ ", actual=" + posNum + ", unseenWord=" + unseenWord;	*/		
				System.out.println(consoleOP);
				bw.write(consoleOP);
				bw.newLine();
				bw.flush();
				if (getPosIndex(output.getData()) == posNum) {
					correctCount++;
				}
				Encog.getInstance().shutdown();

				if (word.equals(sentenceEnd)) {
					prevPOS = START;
				} else {
					prevPOS = pos;
				}
				lineCount++;
			}
			br.close();
			System.out.println("Count : "+correctCount+"/"+lineCount);
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public String getClosestWord (String a1) {
		int minDis = 1000;
		String dreamString = "";
		for (String a2 : inputCorpus) {
			int dist = StringUtils.getLevenshteinDistance(a1, a2);
			if (dist < minDis) {
				dreamString = a2;
				minDis = dist;
			}
		}
		return dreamString;
	}
	
	public String getClosestEnd (String a1) {
		int len = a1.length();
		if (len > 2) {
			len = len-2;
			a1 = a1.substring(len, a1.length());
		}
		String dreamString = "";
		String a3 = "";
		for (String a2 : inputCorpus) {
			len = a2.length();
			if (len > 2) {
				len = len-2;
				a3 = a2.substring(len, a2.length());
			}
			if (a1.equalsIgnoreCase(a3)) 
				dreamString = a2;
		}
		return dreamString;
	}
	public void formCorpus() {

		try {
			File inputfile = new File("resources/mal_pos.train");

			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputfile), "UTF8"));
			String str;
			Set<String> wordSet = new TreeSet<>();
			Set<String> tagSet = new TreeSet<>();
			while ((str = br.readLine()) != null) {
				str = str.trim();
				if ("".equals(str))
					continue;
				StringTokenizer stt = new StringTokenizer(str);
				wordSet.add(stt.nextToken());
				tagSet.add(stt.nextToken());
			}
			br.close();
			String wordCollection = "resources/wordCollection.txt";
			File opFile = new File(wordCollection);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(opFile, true), "UTF-8"));
			for (String word : wordSet) {
				bw.write(word);
				bw.newLine();
				bw.flush();
			}
			bw.close();
			String posCollection = "resources/posCollection.txt";
			opFile = new File(posCollection);
			bw = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(opFile, true), "UTF-8"));
			for (String tag : tagSet) {
				bw.write(tag);
				bw.newLine();
				bw.flush();
			}
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void loadCorpus() {
		try {
			File inputfile = new File("resources/NN_Input.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputfile), "UTF8"));
			String str;
			while ((str = br.readLine()) != null) {
				str = str.trim();
				if ("".equals(str))
					continue;
				inputCorpus.add(str);
			}
			br.close();
			inputfile = new File("resources/posCollection.txt");
			br = new BufferedReader(new InputStreamReader(new FileInputStream(
					inputfile), "UTF8"));
			while ((str = br.readLine()) != null) {
				str = str.trim();
				if ("".equals(str))
					continue;
				posCorpus.add(str);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	public static int getPosIndex(double[] x) {
		double max = 0;
		List<Double> y = new ArrayList<>();
		for (double input : x) {
			y.add(input);
			if (input > max)
				max = input;
		}
		return y.indexOf(max);
	}
	
/*	public static double trainNetwork(final String what,
			final BasicNetwork network, final MLDataSet trainingSet) {
		// train the neural network
		CalculateScore score = new TrainingSetScore(trainingSet);
		final MLTrain trainAlt = new NeuralSimulatedAnnealing(network, score,
				10, 2, 100);

		final MLTrain trainMain = new Backpropagation(network, trainingSet,
				0.01, 0.0);

		final StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
		trainMain.addStrategy(new HybridStrategy(trainAlt));
		// trainMain.addStrategy(stop);

		int epoch = 0;
		while (epoch < 3) {
			trainMain.iteration();
			System.out.println("Training " + what + ", Epoch #" + epoch
					+ " Error:" + trainMain.getError());
			epoch++;
		}
		
		ResilientPropagation prop=new ResilientPropagation(network,trainingSet);
		int epoch = 1;
		do {
			prop.iteration();
			System.out
					.println("Epoch #" + epoch + " Error:" + prop.getError());
			epoch++;
		} while (epoch < 5);
		prop.finishTraining();
		return prop.getError();
	}*/


	public static void main(String[] args) {
		PosTagger tagger = new PosTagger("jordan");	
//		tagger.trainNN();
		tagger.testNN();

	}

}
