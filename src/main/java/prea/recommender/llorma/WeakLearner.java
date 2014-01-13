package prea.recommender.llorma;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * A class learning each local model used in singleton LLORMA.
 * Implementation is based on weighted-SVD.
 * Technical detail of the algorithm can be found in
 * Joonseok Lee and Seungyeon Kim and Guy Lebanon and Yoram Singer, Local Low-Rank Matrix Approximation,
 * Proceedings of the 30th International Conference on Machine Learning, 2013.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 11
 * @version 1.2
 */
public class WeakLearner extends Thread {
	/** The unique identifier of the thread. */
	private int threadId;
	/** The number of features. */
	private int rank;
	/** The number of users. */
	private int userCount;
	/** The number of items. */
	private int itemCount;
	/** The anchor user used to learn this local model. */
	private int anchorUser;
	/** The anchor item used to learn this local model. */
	private int anchorItem;
	/** Learning rate parameter. */
	public double learningRate;
	/** The maximum number of iteration. */
	public int maxIter;
	/** Regularization factor parameter. */
	public double regularizer;
	/** The vector containing each user's weight. */
	private SparseVector w;
	/** The vector containing each item's weight. */
	private SparseVector v; 
	/** User profile in low-rank matrix form. */
	private SparseMatrix userFeatures;
	/** Item profile in low-rank matrix form. */
	private SparseMatrix itemFeatures;
	/** The rating matrix used for learning. */
	private SparseMatrix rateMatrix;
	/** The current train error. */
	private double trainErr;
	
	/**
	 * Construct a local model for singleton LLORMA.
	 * 
	 * @param id A unique thread ID.
	 * @param rk The rank which will be used in this local model.
	 * @param u The number of users.
	 * @param i The number of items.
	 * @param au The anchor user used to learn this local model.
	 * @param ai The anchor item used to learn this local model.
	 * @param lr Learning rate parameter.
	 * @param r Regularization factor parameter.
	 * @param w0 Initial vector containing each user's weight.
	 * @param v0 Initial vector containing each item's weight.
	 * @param rm The rating matrix used for learning.
	 */
	public WeakLearner(int id, int rk, int u, int i, int au, int ai, double lr, double r, int iter, SparseVector w0, SparseVector v0, SparseMatrix rm) {
		threadId = id;
		rank = rk;
		userCount = u;
		itemCount = i;
		anchorUser = au;
		anchorItem = ai;
		learningRate = lr;
		regularizer = r;
		maxIter = iter;
		w = w0;
		v = v0;
		userFeatures = new SparseMatrix(userCount+1, rank);
		itemFeatures = new SparseMatrix(rank, itemCount+1);
		rateMatrix = rm;
	}
	
	/**
	 * Getter method for thread ID.
	 * 
	 * @return The thread ID of this local model.
	 */
	public int getThreadId() {
		return threadId;
	}
	
	/**
	 * Getter method for rank of this local model.
	 * 
	 * @return The rank of this local model.
	 */
	public int getRank() {
		return rank;
	}
	
	/**
	 * Getter method for anchor user of this local model.
	 * 
	 * @return The anchor user ID of this local model.
	 */
	public int getAnchorUser() {
		return anchorUser;
	}
	
	/**
	 * Getter method for anchor item of this local model.
	 * 
	 * @return The anchor item ID of this local model.
	 */
	public int getAnchorItem() {
		return anchorItem;
	}
	
	/**
	 * Getter method for user profile of this local model.
	 * 
	 * @return The user profile of this local model.
	 */
	public SparseMatrix getUserFeatures() {
		return userFeatures;
	}
	
	/**
	 * Getter method for item profile of this local model.
	 * 
	 * @return The item profile of this local model.
	 */
	public SparseMatrix getItemFeatures() {
		return itemFeatures;
	}
	
	/**
	 * Getter method for current train error.
	 * 
	 * @return The current train error.
	 */
	public double getTrainErr() {
		return trainErr;
	}
	
	/** Learn this local model based on similar users to the anchor user
	 * and similar items to the anchor item.
	 * Implemented with gradient descent. */
	@Override
	public void run() {
		//System.out.println("[START] Learning thread " + threadId);
		
		trainErr = Double.MAX_VALUE;
		boolean showProgress = false;
		
		for (int u = 1; u <= userCount; u++) {
			for (int r = 0; r < rank; r++) {
				double rdm = Math.random();
				userFeatures.setValue(u, r, rdm);
			}
		}
		for (int i = 1; i <= itemCount; i++) {
			for (int r = 0; r < rank; r++) {
				double rdm = Math.random();
				itemFeatures.setValue(r, i, rdm);
			}
		}
		
		// Learn by Weighted RegSVD
		int round = 0;
		int rateCount = rateMatrix.itemCount();
		double prevErr = 99999;
		double currErr = 9999;
		
		while (Math.abs(prevErr - currErr) > 0.0001 && round < maxIter) {
			double sum = 0.0;
			for (int u = 1; u <= userCount; u++) {
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						double RuiEst = 0.0;
						for (int r = 0; r < rank; r++) {
							RuiEst += userFeatures.getValue(u, r) * itemFeatures.getValue(r, i);
						}
						double RuiReal = rateMatrix.getValue(u, i);
						double err = RuiReal - RuiEst;
						sum += Math.pow(err, 2);
						
						double weight = w.getValue(u) * v.getValue(i);
						
						for (int r = 0; r < rank; r++) {
							double Fus = userFeatures.getValue(u, r);
							double Gis = itemFeatures.getValue(r, i);
							
							userFeatures.setValue(u, r, Fus + learningRate*(err*Gis*weight - regularizer*Fus));
							if(Double.isNaN(Fus + learningRate*(err*Gis*weight - regularizer*Fus))) {
								System.out.println("a");
							}
							itemFeatures.setValue(r, i, Gis + learningRate*(err*Fus*weight - regularizer*Gis));
							if(Double.isNaN(Gis + learningRate*(err*Fus*weight - regularizer*Gis))) {
								System.out.println("b");
							}
						}
					}
				}
			}
			
			prevErr = currErr;
			currErr = sum/rateCount;
			trainErr = Math.sqrt(currErr);
			
			round++;
			
			// Show progress:
			if (showProgress) {
				System.out.println(round + "\t" + currErr);
			}
		}
		
		//System.out.println("[END] Learning thread " + threadId);
	}
}