package prea.recommender.matrix;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.Distribution;

/**
 * This is a class implementing Bayesian Probabilistic Matrix Factorization.
 * Technical detail of the algorithm can be found in
 * Ruslan Salakhutdinov and Andriy Mnih, Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo,
 * Proceedings of the 25th International Conference on Machine Learning, 2008.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class BayesianPMF extends MatrixFactorizationRecommender {
	private static final long serialVersionUID = 4004;
	
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
	 * @param fc The number of features used for describing user and item profiles.
	 * @param lr Learning rate for gradient-based or iterative optimization.
	 * @param r Controlling factor for the degree of regularization. 
	 * @param m Momentum used in gradient-based or iterative optimization.
	 * @param iter The maximum number of iterations.
	 * @param verbose Indicating whether to show iteration steps and train error.
	 */
	public BayesianPMF(int uc, int ic, double max, double min, int fc, double lr, double r, double m, int iter, boolean verbose) {
		super(uc, ic, max, min, fc, lr, r, m, iter, verbose);
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
		
		double prevErr = 99999999;
		
		// Initialize hierarchical priors:
		int beta = 2; // observation noise (precision)
		SparseVector mu_u = new SparseVector(featureCount);
		SparseVector mu_m = new SparseVector(featureCount);
		SparseMatrix alpha_u = SparseMatrix.makeIdentity(featureCount);
		SparseMatrix alpha_m = SparseMatrix.makeIdentity(featureCount);
		
		// parameters of Inv-Whishart distribution:
		SparseMatrix WI_u = SparseMatrix.makeIdentity(featureCount);
		int b0_u = 2;
		int df_u = featureCount;
		SparseVector mu0_u = new SparseVector(featureCount);
		
		SparseMatrix WI_m = SparseMatrix.makeIdentity(featureCount);
		int b0_m = 2;
		int df_m = featureCount;
		SparseVector mu0_m = new SparseVector(featureCount);
		
		double mean_rating = rateMatrix.average();
		this.offset = mean_rating;
		
		// Initialization using MAP solution found by PMF:
		for (int f = 0; f < featureCount; f++) {
			for (int u = 1; u <= userCount; u++) {
				userFeatures.setValue(u, f, Distribution.normalRandom(0, 1));
			}
			for (int i = 1; i <= itemCount; i++) {
				itemFeatures.setValue(f, i, Distribution.normalRandom(0, 1));
			}
		}
		
		for (int f = 0; f < featureCount; f++) {
			mu_u.setValue(f, userFeatures.getColRef(f).average());
			mu_m.setValue(f, itemFeatures.getRowRef(f).average());
		}
		alpha_u = (userFeatures.covariance()).inverse();
		alpha_m = (itemFeatures.transpose().covariance()).inverse();


		// Iteration:
		SparseVector x_bar = new SparseVector(featureCount);
		SparseVector normalRdn = new SparseVector(featureCount);
		SparseMatrix S_bar, WI_post, lam;
		SparseVector mu_temp;
		double df_upost, df_mpost;
		
		for (int round = 1; round <= maxIter; round++) {
			// Sample from user hyper parameters:
			int M = userCount;
			
			for (int f = 0; f < featureCount; f++) {
				x_bar.setValue(f, userFeatures.getColRef(f).average());
			}
			S_bar = userFeatures.covariance();

			SparseVector mu0_u_x_bar = mu0_u.minus(x_bar);
			SparseMatrix e1e2 = mu0_u_x_bar.outerProduct(mu0_u_x_bar).scale((double) M * (double) b0_u / (double) (b0_u + M));
			WI_post = WI_u.inverse().plus(S_bar.scale(M)).plus(e1e2);
			WI_post = WI_post.inverse();
			WI_post = (WI_post.plus(WI_post.transpose())).scale(0.5);
			
			df_upost = df_u + M;
			SparseMatrix wishrnd_u = Distribution.wishartRandom(WI_post, df_upost);
			if (wishrnd_u != null)
				alpha_u = wishrnd_u; 
			mu_temp = ((mu0_u.scale(b0_u)).plus(x_bar.scale(M))).scale(1 / ((double) b0_u + (double) M));
			lam = alpha_u.scale(b0_u + M).inverse().cholesky();
			
			if (lam != null) {
				lam = lam.transpose();
				
				normalRdn = new SparseVector(featureCount);
				for (int f = 0; f < featureCount; f++) {
					normalRdn.setValue(f, Distribution.normalRandom(0, 1));
				}
				
				mu_u = lam.times(normalRdn).plus(mu_temp);
			}
			
			//Sample from item hyper parameters:  
			int N = itemCount;
			
			for (int f = 0; f < featureCount; f++) {
				x_bar.setValue(f, itemFeatures.getRowRef(f).average());
			}
			S_bar = itemFeatures.transpose().covariance();

			SparseVector mu0_m_x_bar = mu0_m.minus(x_bar);
			SparseMatrix e3e4 = mu0_m_x_bar.outerProduct(mu0_m_x_bar).scale((double) N * (double) b0_m / (double) (b0_m + N));
			WI_post = WI_m.inverse().plus(S_bar.scale(N)).plus(e3e4);
			WI_post = WI_post.inverse();
			WI_post = (WI_post.plus(WI_post.transpose())).scale(0.5);
			
			df_mpost = df_m + N;
			SparseMatrix wishrnd_m = Distribution.wishartRandom(WI_post, df_mpost);
			if (wishrnd_m != null)
				alpha_m = wishrnd_m;
			mu_temp = ((mu0_m.scale(b0_m)).plus(x_bar.scale(N))).scale(1 / ((double) b0_m + (double) N));
			lam = alpha_m.scale(b0_m + N).inverse().cholesky();
			
			if (lam != null) {
				lam = lam.transpose();
			
				normalRdn = new SparseVector(featureCount);
				for (int f = 0; f < featureCount; f++) {
					normalRdn.setValue(f, Distribution.normalRandom(0, 1));
				}
				
				mu_m = lam.times(normalRdn).plus(mu_temp);
			}
			
			// Gibbs updates over user and item feature vectors given hyper parameters:
			for (int gibbs = 1; gibbs < 2; gibbs++) {
				// Infer posterior distribution over all user feature vectors 
				for (int uu = 1; uu <= userCount; uu++) {
					// list of items rated by user uu:
					int[] ff = rateMatrix.getRowRef(uu).indexList();
					
					if (ff == null)
						continue;
					
					int ff_idx = 0;
					for (int t = 0; t < ff.length; t++) {
						ff[ff_idx] = ff[t];
						ff_idx++;
					}
					
					// features of items rated by user uu:
					SparseMatrix MM = new SparseMatrix(ff_idx, featureCount);
					SparseVector rr = new SparseVector(ff_idx);
					int idx = 0;
					for (int t = 0; t < ff_idx; t++) {
						int i = ff[t];
						rr.setValue(idx, rateMatrix.getValue(uu, i) - mean_rating);
						for (int f = 0; f < featureCount; f++) {
							MM.setValue(idx, f, itemFeatures.getValue(f, i));
						}
						idx++;
					}
					
					SparseMatrix covar = (alpha_u.plus((MM.transpose().times(MM)).scale(beta))).inverse();
					SparseVector a = MM.transpose().times(rr).scale(beta);
					SparseVector b = alpha_u.times(mu_u);
					SparseVector mean_u = covar.times(a.plus(b));
					lam = covar.cholesky();
					
					if (lam != null) {
						lam = lam.transpose();
						for (int f = 0; f < featureCount; f++) {
							normalRdn.setValue(f, Distribution.normalRandom(0, 1));
						}
						
						SparseVector w1_P1_uu = lam.times(normalRdn).plus(mean_u);
						
						for (int f = 0; f < featureCount; f++) {
							userFeatures.setValue(uu, f, w1_P1_uu.getValue(f));
						}
					}
				}
				
				// Infer posterior distribution over all movie feature vectors 
				for (int ii = 1; ii <= itemCount; ii++) {
					// list of users who rated item ii:
					int[] ff = rateMatrix.getColRef(ii).indexList();
					
					if (ff == null)
						continue;
					
					int ff_idx = 0;
					for (int t = 0; t < ff.length; t++) {
						ff[ff_idx] = ff[t];
						ff_idx++;
					}
					
					// features of users who rated item ii:
					SparseMatrix MM = new SparseMatrix(ff_idx, featureCount);
					SparseVector rr = new SparseVector(ff_idx);
					int idx = 0;
					for (int t = 0; t < ff_idx; t++) {
						int u = ff[t];
						rr.setValue(idx, rateMatrix.getValue(u, ii) - mean_rating);
						for (int f = 0; f < featureCount; f++) {
							MM.setValue(idx, f, userFeatures.getValue(u, f));
						}
						idx++;
					}
					
					SparseMatrix covar = (alpha_m.plus((MM.transpose().times(MM)).scale(beta))).inverse();
					SparseVector a = MM.transpose().times(rr).scale(beta);
					SparseVector b = alpha_m.times(mu_m);
					SparseVector mean_m = covar.times(a.plus(b));
					lam = covar.cholesky();
					
					if (lam != null) {
						lam = lam.transpose();
						for (int f = 0; f < featureCount; f++) {
							normalRdn.setValue(f, Distribution.normalRandom(0, 1));
						}
						
						SparseVector w1_M1_ii = lam.times(normalRdn).plus(mean_m);
						
						for (int f = 0; f < featureCount; f++) {
							itemFeatures.setValue(f, ii, w1_M1_ii.getValue(f));
						}
					}
				}
			}
			
			
			// show progress:
			double err = 0.0;
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					for (int i : itemList) {
						double Aij = rateMatrix.getValue(u, i) - mean_rating;
						double Bij = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
						err += Math.pow(Aij - Bij, 2);
					}
				}
			}
			
			if (showProgress) {
				System.out.println(round + "\t" + (err / rateMatrix.itemCount()));
			}
			
			if (prevErr < err) {
				break;
			}
			else {
				prevErr = err;
			}
		}
	}
}
