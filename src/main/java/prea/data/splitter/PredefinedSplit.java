package prea.data.splitter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import prea.data.structure.SparseMatrix;

/**
 * When a predefined split file is available,
 * this class helps to split train/test set as defined in the file.
 * This can be used for verifying implementation of new CF algorithm. 
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class PredefinedSplit extends DataSplitManager {
	/*========================================
	 * Constructors
	 *========================================*/
	/** Construct an instance for splitter with predefined split file. */
	public PredefinedSplit(SparseMatrix originalMatrix, String splitFileName, int max, int min) {
		super(originalMatrix, max, min);
		readSplitData(splitFileName);
		calculateAverage((maxValue + minValue) / 2);
	}
	
	/**
	 * Split the rating matrix into train and test set, by given split data file.
	 * 
	 * @param fileName the name of split data file. 
	 * 
	 **/
	private void readSplitData(String fileName) {
		recoverTestItems();
		
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			// Read Train/Test user/item data:
			String line;
			int testUserCount = userCount;
			boolean[] isTestUser = new boolean[userCount+1];
			int[] testUserList = new int[testUserCount];
			
			for (int u = 0; u < userCount; u++) {
				isTestUser[u+1] = true;
				testUserList[u] = u+1;
			}
			
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				StringTokenizer st = new StringTokenizer (line);
				int userNo = Integer.parseInt(st.nextToken());
				int itemNo = Integer.parseInt(st.nextToken());
				isTestUser[userNo] = true;
				
				testMatrix.setValue(userNo, itemNo, rateMatrix.getValue(userNo, itemNo));
				rateMatrix.setValue(userNo, itemNo, 0.0);
			}
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + fileName);
			return;
		}
	}
}
