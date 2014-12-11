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

import org.encog.util.file.FileUtil;

public class ProBenRunner {
	private File dir;
	private String methodName;
	private String trainingName;
	private String methodArchitecture;
	private String trainingArgs;
	
	public ProBenRunner(File theDir, String theMethodName, String theTrainingName, String theMethodArchitecture, String theTrainingArgs) {
		this.dir = theDir;
		this.methodName = theMethodName;
		this.trainingName = theTrainingName;
		this.methodArchitecture = theMethodArchitecture;
		this.trainingArgs = theTrainingArgs;
	}
	
	public void run() {
		runDirectory(this.dir);
	}
	
	public void runDirectory(File file) {
		
		for(File childFile: file.listFiles()) {
			if( childFile.isDirectory()) {
				runDirectory(childFile);				
			} else {
				if( FileUtil.getFileExt(childFile).equalsIgnoreCase("dt")) {
					runFile(childFile);
				}
			}
		}
		
	}
	
	public void runFile(File file) {
		ProBenData data = new ProBenData(file);
		data.load();
		
		ProBenEvaluate eval = new ProBenEvaluate(data, methodName,trainingName,methodArchitecture,trainingArgs);
		eval.evaluate();
	}
}
