package prea.recommender.matrix;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * This is a class implementing Non-negative Matrix Factorization.
 * Technical detail of the algorithm can be found in
 * Daniel D. Lee and H. Sebastian Seung, Algorithms for Non-negative Matrix Factorization, Advances in Neural Information Processing Systems, 2001.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class NMF extends MatrixFactorizationRecommender {
	private static final long serialVersionUID = 4002;
	
	/** Rating matrix for items which will be used during the validation phase.
	 * Not allowed to refer during training phase. */
	private SparseMatrix validationMatrix;
	/** Proportion of dataset, using for validation purpose. */
	private double validationRatio;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features in low-rank factorized matrix.
	 * @param lr The learning rate for gradient-descent method.
	 * @param r The regularization factor.
	 * @param m The momentum parameter.
	 * @param iter The maximum number of iteration.
	 * @param verbose Show progress of iterative methods.
	 */
	public NMF(int uc, int ic, double max, double min, int fc, double lr, double r, double m, int iter, double vr, boolean verbose) {
		super(uc, ic, max, min, fc, lr, r, m, iter, verbose);
		validationRatio = vr;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rateMatrix Training data set.
	 */
	@Override
	public void buildModel(SparseMatrix rateMatrix) {
		super.buildModel(rateMatrix);
		
		makeValidationSet(rateMatrix, validationRatio);
		
		int round = 0;
		double prevErr = 99999;
		double currErr = 9999;
		
		while (prevErr > currErr && round < maxIter) {
			// User features update:
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					SparseVector ratedItems = new SparseVector(itemCount+1);
					for (int i : itemList) {
						double estimate = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
						ratedItems.setValue(i, estimate);
					}
					
					for (int f = 0; f < featureCount; f++) {
						double estimatedProfile = ratedItems.innerProduct(itemFeatures.getRowRef(f));
						double realProfile = rateMatrix.getRowRef(u).innerProduct(itemFeatures.getRowRef(f));
						double ratio = Math.max(realProfile - regularizer, 1E-9) / (estimatedProfile + 1E-9);
						
						userFeatures.setValue(u, f, userFeatures.getValue(u, f) * ratio);
					}
				}
			}
			
			// Item features update:
			for (int i = 1; i <= itemCount; i++) {
				int[] userList = rateMatrix.getColRef(i).indexList();
				
				if (userList != null) {
					SparseVector ratedUsers = new SparseVector(userCount+1);
					for (int u : userList) {
						double estimate = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
						ratedUsers.setValue(u, estimate);
					}
					
					for (int f = 0; f < featureCount; f++) {
						double estimatedProfile = userFeatures.getColRef(f).innerProduct(ratedUsers);
						double realProfile = userFeatures.getColRef(f).innerProduct(rateMatrix.getColRef(i));
						double ratio = Math.max(realProfile - regularizer, 1E-9) / (estimatedProfile + 1E-9);
						
						itemFeatures.setValue(f, i, itemFeatures.getValue(f, i) * ratio);
					}
				}
			}
			
			round++;
			
			// show progress:
			double err = 0.0;
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = validationMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					for (int i : itemList) {
						double Aij = validationMatrix.getValue(u, i);
						double Bij = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
						err += Math.pow(Aij - Bij, 2);
					}
				}
			}
			
			prevErr = currErr;
			currErr = err/validationMatrix.itemCount();
			
			if (showProgress)
				System.out.println(round + "\t" + Math.sqrt(currErr));
		}
		
		restoreValidationSet(rateMatrix);
	}
	
	/**
	 * Items which will be used for validation purpose are moved from rateMatrix to validationMatrix.
	 * 
	 * @param validationRatio Proportion of dataset, using for validation purpose.
	 */
	private void makeValidationSet(SparseMatrix rateMatrix, double validationRatio) {
		validationMatrix = new SparseMatrix(userCount+1, itemCount+1);
		
		int validationCount = (int) (rateMatrix.itemCount() * validationRatio);
		while (validationCount > 0) {
			int index = (int) (Math.random() * userCount) + 1;
			SparseVector row = rateMatrix.getRowRef(index);
			int[] itemList = row.indexList();
			
			if (itemList != null && itemList.length > 5) {
				int index2 = (int) (Math.random() * itemList.length);
				validationMatrix.setValue(index, itemList[index2], rateMatrix.getValue(index, itemList[index2]));
				rateMatrix.setValue(index, itemList[index2], 0.0);
				
				validationCount--;
			}
		}
	}
	
	/** Items in validationMatrix are moved to original rateMatrix. */
	private void restoreValidationSet(SparseMatrix rateMatrix) {
		for (int i = 1; i <= userCount; i++) {
			SparseVector row = validationMatrix.getRowRef(i);
			int[] itemList = row.indexList();
			
			if (itemList != null) {
				for (int j : itemList) {
					rateMatrix.setValue(i, j, validationMatrix.getValue(i, j));
				}
			}
		}
	}
}
