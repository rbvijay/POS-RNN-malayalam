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
package org.encog.examples.proben;

import java.io.File;

import org.encog.Encog;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.factory.MLTrainFactory;

public class ProBen {
	
	public final static String METHOD_NAME = MLMethodFactory.TYPE_FEEDFORWARD;
	public final static String TRAINING_TYPE = MLTrainFactory.TYPE_PSO;
	public final static String METHOD_ARCHITECTURE = "?:B->SIGMOID->40:B->SIGMOID->?";
	public final static String TRAINING_ARGS = "";
	
	
	public static void main(String[] args) {
		
		try {
			ProBenRunner runner = new ProBenRunner(new File("C:\\test\\proben1\\"),
					METHOD_NAME,
					TRAINING_TYPE,
					METHOD_ARCHITECTURE,
					TRAINING_ARGS);
			
			runner.run();

			Encog.getInstance().shutdown();
			
			
		} catch(Exception ex) {
			ex.printStackTrace();
		}
		
	}
}
