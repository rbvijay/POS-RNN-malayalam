/*
 * Encog(tm) Java Examples v3.3
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-examples
 *
 * Copyright 2008-2014 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package org.encog.examples.neural.predict.market;

import java.io.File;

import org.encog.Encog;



/**
 * Use the saved market neural network, and now attempt to predict for today, and the
 * last 60 days and see what the results are.
 */
public class MarketPredict {
		
	public static void main(String[] args)
	{		
		if( args.length<1 ) {
			System.out.println("MarketPredict [data dir] [generate/train/incremental/evaluate]");
		}
		else
		{
			File dataDir = new File(args[0]);
			if( args[1].equalsIgnoreCase("generate") ) {
				MarketBuildTraining.generate(dataDir);
			} 
			else if( args[1].equalsIgnoreCase("train") ) {
				MarketTrain.train(dataDir);
			} 
			else if( args[1].equalsIgnoreCase("evaluate") ) {
				MarketEvaluate.evaluate(dataDir);
			} else if( args[1].equalsIgnoreCase("prune") ) {
				MarketPrune.incremental(dataDir);
			} 
			Encog.getInstance().shutdown();
		}
	}
	
}
