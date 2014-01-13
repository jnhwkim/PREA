package prea.recommender.memory;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.Recommender;

/**
 * The class implementing two memory-based (neighborhood-based) methods,
 * predicting by referring to rating matrix for each query.
 * Contains user-based and item-based, with some variations on them.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public abstract class MemoryBasedRecommender implements Recommender {
	// similarity measure
	/** Similarity Measure Code for Pearson Correlation */
	public static final int PEARSON_CORR = 101;
	/** Similarity Measure Code for Vector Cosine */
	public static final int VECTOR_COS = 102;
	/** Similarity Measure Code for Mean Squared Difference (MSD) */
	public static final int MEAN_SQUARE_DIFF = 103;
	/** Similarity Measure Code for Mean Absolute Difference (MAD) */
	public static final int MEAN_ABS_DIFF = 104;
	/** Similarity Measure Code for Inverse User Frequency */
	public static final int INVERSE_USER_FREQUENCY = 105;
	
	// estimation
	/** Estimation Method Code for Weighted Sum */
	public static final int WEIGHTED_SUM = 201;
	/** Estimation Method Code for Simple Weighted Average */
	public static final int SIMPLE_WEIGHTED_AVG = 202;
	
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public int minValue;
	
	/** The number of neighbors, used for estimation. */
	public int neighborSize;
	/** The method code for similarity measure. */
	public int similarityMethod;
	
	/** Indicating whether to use default vote value. */
	public boolean defaultVote;
	/** The default voting value, if used. */
	public double defaultValue;
	
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a memory-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param ns The neighborhood size.
	 * @param sim The method code of similarity measure.
	 * @param df Indicator whether to use default values.
	 * @param dv Default value if used.
	 */
	public MemoryBasedRecommender(int uc, int ic, int max, int min, int ns, int sim, boolean df, double dv) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		
		neighborSize = ns;
		similarityMethod = sim;
		
		defaultVote = df;
		defaultValue = dv;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rm Training data set.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		rateMatrix = rm;
	}
	
	/*========================================
	 * Possible options
	 *========================================*/
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
	public double similarity(boolean rowOriented, SparseVector i1, SparseVector i2, double i1Avg, double i2Avg, int method) {
		double result = 0.0;
		SparseVector v1, v2;
		
		if (defaultVote) {
			int[] i1ItemList = i1.indexList();
			int[] i2ItemList = i2.indexList();
			v1 = new SparseVector(i1.length());
			v2 = new SparseVector(i2.length());
			
			if (i1ItemList != null) {
				for (int t = 0; t < i1ItemList.length; t++) {
					v1.setValue(i1ItemList[t], i1.getValue(i1ItemList[t]));
					if (i2.getValue(i1ItemList[t]) == 0.0) {
						v2.setValue(i1ItemList[t], defaultValue);
					}
				}
			}
			
			if (i2ItemList != null) {
				for (int t = 0; t < i2ItemList.length; t++) {
					v2.setValue(i2ItemList[t], i2.getValue(i2ItemList[t]));
					if (i1.getValue(i2ItemList[t]) == 0.0) {
						v1.setValue(i2ItemList[t], defaultValue);
					}
				}
			}
		}
		else {
			v1 = i1;
			v2 = i2;
		}
		
		if (method == PEARSON_CORR) { // Pearson correlation
			SparseVector a = v1.sub(i1Avg);
			SparseVector b = v2.sub(i2Avg);
			
			result = a.innerProduct(b) / (a.norm() * b.norm());
		}
		else if (method == VECTOR_COS) { // Vector cosine
			result = v1.innerProduct(v2) / (v1.norm() * v2.norm());
		}
		else if (method == MEAN_SQUARE_DIFF) { // Mean Square Difference
			SparseVector a = v1.commonMinus(v2);
			a = a.power(2);
			result = a.sum() / a.itemCount();
		}
		else if (method == MEAN_ABS_DIFF) { // Mean Absolute Difference
			SparseVector a = v1.commonMinus(v2);
			result = a.absoluteSum() / a.itemCount();
		}
		else if (method == INVERSE_USER_FREQUENCY) {
			SparseVector a = v1.commonMinus(v2);
			int[] commonItemList = a.indexList();
			
			if (commonItemList == null)
				return 0.0;
			
			double invFreqSum = 0.0;
			double invFreqUser1Sum = 0.0;
			double invFreqUser2Sum = 0.0;
			double invFreqUser11Sum = 0.0;
			double invFreqUser22Sum = 0.0;
			double invFreqUser12Sum = 0.0;
			
			for (int t = 0; t < commonItemList.length; t++) {
				double invFreq = Math.log(userCount / rateMatrix.getColRef(commonItemList[t]).itemCount());
				
				invFreqSum += invFreq;
				invFreqUser1Sum += (invFreq * v1.getValue(commonItemList[t]));
				invFreqUser2Sum += (invFreq * v2.getValue(commonItemList[t]));
				invFreqUser11Sum += (invFreq * v1.getValue(commonItemList[t]) * v1.getValue(commonItemList[t]));
				invFreqUser22Sum += (invFreq * v1.getValue(commonItemList[t]) * v2.getValue(commonItemList[t]));
				invFreqUser12Sum += (invFreq * v1.getValue(commonItemList[t]) * v2.getValue(commonItemList[t]));
			}
			
			result = (invFreqSum * invFreqUser12Sum - invFreqUser1Sum * invFreqUser2Sum)
					/ Math.sqrt(invFreqSum * (invFreqUser11Sum - invFreqUser1Sum * invFreqUser1Sum)
											* (invFreqUser22Sum - invFreqUser2Sum * invFreqUser2Sum));
		}
		
		return result;
	}
}
