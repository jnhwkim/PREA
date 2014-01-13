package prea.main;

import java.io.*;
import java.util.*;
import prea.data.structure.*;

/**
 * This class helps to save train/test split and similarity prefetch files.
 * This may be used for repeated experiment on same environment. 
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Splitter {
	// parameters
	/** Ratio of dataset which will be used for test purpose. */
	public static double testRatio = 0.2;
	
	// similarity measure
	/** Similarity measure code for Pearson Correlation. */
	public static final int PEARSON_CORR = 101;
	/** Similarity measure code for Vector Cosine. */
	public static final int VECTOR_COS = 102;
	/** Similarity measure code for Mean Squared Difference. */
	public static final int MEAN_SQUARE_DIFF = 103;
	/** Similarity measure code for Mean Absoulte Difference. */
	public static final int MEAN_ABS_DIFF = 104;
	
	/** Rating matrix for train dataset. */
	public static SparseMatrix rateMatrix;
	/** Average rating for each user. */
	public static SparseVector userRateAverage;
	/** Average rating for each item. */
	public static SparseVector itemRateAverage;
	/** The list of item names, provided with the dataset. */
	public static String[] columnName;
	
	/** The number of users. */
	public static int userCount;
	/** The number of items. */
	public static int itemCount;
	
	/**
	 * Main method for reading the arff file, writing split and similarity results.
	 * 
	 * @param argv The argument list. First two are required: input file name and testset ratio.
	 * Next two are optional, indicating whether it computes and prints similarity for users and items.
	 */
	public static void main(String argv[]) {
		try {
			boolean computeUserSimilarity = false;
			boolean computeItemSimilarity = false;
			
			// Read arff file
			String readFileName;
			if (argv.length > 2) {
				for (int i = 2; i < argv.length; i++) {
					if (argv[i].equals("-u"))
						computeUserSimilarity = true;
					if (argv[i].equals("-i"))
						computeItemSimilarity = true;
				}
			}
			if (argv.length > 1) {
				readFileName = argv[0];
				testRatio = Double.parseDouble(argv[1]);
			}
			else {
				System.out.println("Usage: java Splitter [Input File Name] [Testset Ratio] [(Optional) -u] [(Optional) -i]");
				System.out.println("Input File Name and Testset Ratio are required.");
				System.out.println("-u: compute and save user similarity.");
				System.out.println("-i: compute and save item similarity.");
				return;
			}
			
			System.out.println("[START]\tRead arff input file.");
			readArff (readFileName + ".arff");
			System.out.println("[END]\tRead arff input file.");
			
			// Train and test split:
			System.out.println("[START]\tWrite train/test split file.");
			FileOutputStream outputStream = new FileOutputStream(readFileName + "_split.txt");
			PrintWriter pSystem = new PrintWriter (outputStream);
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					for (int i : itemList) {
						double rdm = Math.random();
						
						if (rdm < testRatio) {
							pSystem.println(u + "\t" + i);
							rateMatrix.setValue(u, i, 0.0);
						}
					}
				}
			}
			
			pSystem.flush();
			outputStream.close();
			System.out.println("[END]\tWrite train/test split file.");
			
			// Calculate user similarity:
			if (computeUserSimilarity) {
				System.out.println("[START]\tCompute user similarity.");
				FileOutputStream outputStreamUser = new FileOutputStream(readFileName + "_userSim.txt");
				PrintWriter pSystemUser = new PrintWriter (outputStreamUser);
				
				for (int u = 1; u <= userCount; u++) {
					SparseVector user1 = rateMatrix.getRowRef(u);
					
					for (int v = 1; v <= userCount; v++) {
						SparseVector user2 = rateMatrix.getRowRef(v);
						double sim = similarity (false, user1, user2, userRateAverage.getValue(u), userRateAverage.getValue(v), VECTOR_COS);
						
						if (Double.isNaN(sim)) {
							sim = 0.0;
						}
						
						// Write to output file:
						int sim10000 = (int) (10000 * sim);
						pSystemUser.print (sim10000 + "\t");
					}
					
					pSystemUser.println();
					
					if (u%100 == 0 || u == userCount) {
						System.out.println ("\tUser " + u + "/" + userCount + " done.");
					}
				}
			
				pSystemUser.flush();
				outputStreamUser.close();
				System.out.println("[END]\tCompute user similarity.");
			}
			
			// Calculate item similarity:
			if (computeItemSimilarity) {
				System.out.println("[START]\tCompute item similarity.");
				FileOutputStream outputStreamItem = new FileOutputStream(readFileName + "_itemSim.txt");
				PrintWriter pSystemItem = new PrintWriter (outputStreamItem);
				
				for (int i = 1; i <= itemCount; i++) {
					SparseVector item1 = rateMatrix.getColRef(i);
					
					for (int j = 1; j <= itemCount; j++) {
						SparseVector item2 = rateMatrix.getColRef(j);
						double sim = similarity (false, item1, item2, itemRateAverage.getValue(i), itemRateAverage.getValue(j), VECTOR_COS);
						
						if (Double.isNaN(sim)) {
							sim = 0.0;
						}
						
						// Write to output file:
						int sim10000 = (int) (10000 * sim);
						pSystemItem.print (sim10000 + "\t");
					}
					
					pSystemItem.println();
					
					if (i%100 == 0 || i == itemCount) {
						System.out.println ("\tItem " + i + "/" + itemCount + " done.");
					}
				}
				
				pSystemItem.flush();
				outputStreamItem.close();
				System.out.println("[END]\tCompute item similarity.");
			}
		}
		catch (IOException e) {
			System.out.println("No such file.");
		}
	}
	
	/**
	 * Calculate similarity between two given vectors.
	 * 
	 * @param rowOriented Use true if user-based, false if item-based.
	 * @param i1 The first vector to calculate similarity.
	 * @param i2 The second vector to calculate similarity.
	 * @param i1Avg The average of elements in the first vector.
	 * @param i2Avg The average of elements in the second vector.
	 * @param method The code of similarity measure to be used.
	 * It can be one of the following: PEARSON_CORR, VECTOR_COS,
	 * MEAN_SQUARE_DIFF, MEAN_ABS_DIFF, or INVERSE_USER_FREQUENCY.
	 * @return The similarity value between two vectors i1 and i2.
	 */
	private static double similarity(boolean rowOriented, SparseVector i1, SparseVector i2, double i1Avg, double i2Avg, int method) {
		double result = 0.0;
		
		if (method == PEARSON_CORR) { // Pearson correlation
			SparseVector a = i1.sub(i1Avg);
			SparseVector b = i2.sub(i2Avg);
			
			result = a.innerProduct(b) / (a.norm() * b.norm());
		}
		else if (method == VECTOR_COS) { // Vector cosine
			result = i1.innerProduct(i2) / (i1.norm() * i2.norm());
		}
		else if (method == MEAN_SQUARE_DIFF) { // Mean Square Difference
			SparseVector a = i1.commonMinus(i2);
			a = a.power(2);
			result = a.sum() / a.itemCount();
		}
		else if (method == MEAN_ABS_DIFF) { // Mean Absolute Difference
			SparseVector a = i1.commonMinus(i2);
			result = a.absoluteSum() / a.itemCount();
		}
		
		return result;
	}
	
	/**
	 * Read the data file in ARFF format, and store it in rating matrix.
	 * Peripheral information such as max/min values, user/item count are also set in this method.
	 * 
	 * @param fileName The name of data file.
	 */
	private static void readArff (String fileName) {
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			ArrayList<String> tmpColumnName = new ArrayList<String>();
			
			String line;
			int userNo = 0; // sequence number of each user
			int attributeCount = 0;
			
			// Read attributes:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.contains("@ATTRIBUTE")) {
					String name;
					
					line = line.substring(10).trim();
					if (line.charAt(0) == '\'') {
						int idx = line.substring(1).indexOf('\'');
						name = line.substring(1, idx+1);
					}
					else {
						int idx = line.substring(1).indexOf(' ');
						name = line.substring(0, idx+1).trim();
					}
					
					//columnName[lineNo] = name;
					tmpColumnName.add(name);
					attributeCount++;
				}
				else if (line.contains("@RELATION")) {
					// do nothing
				}
				else if (line.contains("@DATA")) {
					// This is the end of attribute section!
					break;
				}
				else if (line.length() <= 0) {
					// do nothing
				}
			}
			
			// Set item count to data structures:
			itemCount = (attributeCount - 1)/2;
			columnName = new String[attributeCount];
			tmpColumnName.toArray(columnName);
			
			int[] itemRateCount = new int[itemCount+1];
			rateMatrix = new SparseMatrix(500000, itemCount+1); // max 480189, 17770
			userRateAverage = new SparseVector(500000);
			itemRateAverage = new SparseVector(itemCount+1);
			
			// Read data:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.length() > 0) {
					line = line.substring(1, line.length() - 1);
					
					StringTokenizer st = new StringTokenizer (line, ",");
					
					double rateSum = 0.0;
					int rateCount = 0;
					while (st.hasMoreTokens()) {
						String token = st.nextToken().trim();
						int i = token.indexOf(" ");
						
						int movieID, rate;
						int index = Integer.parseInt(token.substring(0, i));
						String data = token.substring(i+1);
						
						if (index == 0) { // User ID
							rateSum = 0.0;
							rateCount = 0;
							
							userNo++;
						}
						else if (data.length() == 1) { // Rate
							movieID = index;
							rate = Integer.parseInt(data);
							
							rateSum += rate;
							rateCount++;
							
							rateMatrix.setValue(userNo, movieID, rate);
							userRateAverage.setValue(userNo, rateSum / rateCount);
							
							itemRateAverage.setValue(movieID, itemRateAverage.getValue(movieID) + rate);
							(itemRateCount[movieID])++;
						}
						else { // Date
							// This part may be used to create time-dependent train/test split file.
							
//							index = index - itemCount;
//							int year = Integer.parseInt(data.substring(0, 4));
//							int month = Integer.parseInt(data.substring(5, 7));
//							
//							if (year >= 2006 || (year == 2005 && month >= 12)) {
//								System.out.println(userNo + "\t" + index + "\t" + data);
//							}
						}
					}
				}
			}
			
			// Item average calculation:
			for (int i = 0; i < itemCount; i++) {
				itemRateAverage.setValue(i, itemRateAverage.getValue(i) / itemRateCount[i]);
			}
			
			userCount = userNo;
			
			System.out.println ("\tUser Count\t" + userCount);
			System.out.println ("\tItem Count\t" + itemCount);
			System.out.println ("\tRating Count\t" + rateMatrix.itemCount());
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file.");
		}
	}
}