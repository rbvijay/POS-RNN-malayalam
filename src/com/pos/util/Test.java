package com.pos.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.lang.StringUtils;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.util.text.LevenshteinDistance;

import com.pos.stringSimilarity.JaroWinklerStrategy;
import com.pos.stringSimilarity.LevenshteinDistanceStrategy;
import com.pos.stringSimilarity.SimilarityStrategy;
import com.pos.stringSimilarity.StringSimilarityService;
import com.pos.stringSimilarity.StringSimilarityServiceImpl;

public class Test {

	static List<String> inputCorpus = new ArrayList<>();
	
	public static void main(String[] args) {
		loadCorpus();
		try {
			File inputfile = new File("resources/mal_pos.test");
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(inputfile), "UTF8"));
			String str;
			while ((str = br.readLine()) != null) {
				str = str.trim();
				if ("".equals(str))
					continue;
				StringTokenizer stt = new StringTokenizer(str);
				String word = stt.nextToken();
				System.out.println(word+" ---- "+getClosestEnd(word));
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

/*		SimilarityStrategy strategy = new LevenshteinDistanceStrategy();
		String target = "ser";
		String source = "antenna";
		StringSimilarityService service = new StringSimilarityServiceImpl(strategy);
		double score = service.score(source, target);
		System.out.println(dist.computeDistance(source, target));;
		System.out.println(score);
		System.out.println(StringUtils.getLevenshteinDistance(source, target));*/
		
	}
	
	public static String getClosestWord (String a1) {
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
	
	public static String getClosestEnd (String a1) {
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
	
	public static void loadCorpus() {
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
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void getNearestWord () {
		
		StringUtils.getLevenshteinDistance("abcd", "xvdf");
	}
}
