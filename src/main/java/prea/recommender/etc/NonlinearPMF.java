package prea.recommender.etc;
import prea.data.structure.DenseMatrix;
import prea.data.structure.DenseVector;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.Recommender;
import prea.util.Distribution;
import prea.util.EvaluationMetrics;
import prea.util.Sort;

/**
 * This is a class implementing Non-linear Probabilistic Matrix Factorization.
 * Technical detail of the algorithm can be found in
 * Neil D. Lawrence and Raquel Urtasun, Non-linear Matrix Factorization with Gaussian Processes,
 * Proceedings of the 26th International Conference on Machine Learning, 2009.
 * 
 * @author Mingxuan Sun
 * @since 2012. 4. 20
 * @version 1.1
 */
public class NonlinearPMF implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	// Transform Type
	public static final int ATOX = 8501;
	public static final int XTOA = 8502;
	public static final int GRADFACT = 8503;
	
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** The number of features. */
	public int featureCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	/** Learning rate parameter. */
	public double learningRate;
	/** Momentum parameter. */
	public double momentum;
	/** Maximum number of iteration. */
	public int maxIter;

	/** Item profile in low-rank matrix form. */
	public SparseMatrix itemFeatures;
	/** Change of Item profile in low-rank matrix form. */
	public SparseMatrix itemFeaturesChange;
	/** Indicator whether to show progress of iteration. */
	public boolean showProgress = false;
	/** kernel parameter number */
	public int kernParamNum;
	/** kernel parameter value */
	public DenseVector kernParam;

	/** list of number from 1 to n */
	int[] index1toAll;

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
	 * @param iter The maximum number of iteration.
	 * @param kernInverseWidth the kernel inverse width for the RBF kernel.
	 * @param kernVarianceRbf the variance for the RBF kernel.
	 * @param kernVarianceBias the variance for the bias.
	 * @param kernVarianceWhite the variance for the white noise.
	 */
	public NonlinearPMF(int uc, int ic, double max, double min, int fc, double l, double m, int iter, 
			double kernInverseWidth, double kernVarianceRbf, double kernVarianceBias, double kernVarianceWhite) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		featureCount = fc;
		learningRate = l; //default 0.0001;
		momentum = m; //default 0.9;
		maxIter = iter;

		index1toAll = new int[featureCount];
		for (int i = 0; i < featureCount; i++){
			index1toAll[i] = i;
		}

		itemFeatures = new SparseMatrix(itemCount, featureCount);
		itemFeaturesChange = new SparseMatrix(itemCount, featureCount);	 
		kernParamNum = 4;
		kernParam = new DenseVector(kernParamNum);
		kernParam.setValue(0, kernInverseWidth); //default 1.0;	 
		kernParam.setValue(1, kernVarianceRbf); //default 1.0;	   
		kernParam.setValue(2, kernVarianceBias); //default 0.11;	  
		kernParam.setValue(3, kernVarianceWhite); //default 5.0;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/** 
	 * Build a model with the given data and algorithm.
	 * 
	 * @param rm The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		rateMatrix = rm;
		
		// Initialize item features randomly:
		for (int i = 0; i < itemCount; i++) {
			for (int f = 0; f < featureCount; f++) {
				double rdm = 0.001 * Distribution.normalRandom(0, 1);
				itemFeatures.setValue(i, f, rdm);
			}
		}

		// Optimization options:
		boolean optimiseParam = true;
		DenseVector param = expTransform(kernParam, XTOA);

		// Gradient of parameters:
		DenseVector changeParam = new DenseVector(kernParamNum);    	   
		int startUser = 1;
		int startIter = 1;

		for (int iters = startIter; iters <= maxIter; iters++) {
			for (int user = startUser; user <= userCount; user++) {
				if (rateMatrix.getRowRef(user).itemCount() > 0) {
					DenseVector g_param = collabLogLikeGradients(user,kernParam, momentum, learningRate);

					if(optimiseParam) {	
						changeParam = (changeParam.scale(momentum)).plus(g_param.scale(learningRate));
						param = param.plus(changeParam); 
						kernParam = expTransform(param, ATOX); //change kern variable here				     
					}
				}
			}
		}
	}

	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	@Override
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);		
		for (int u = 1; u <= userCount; u++) {		 
			int[] testItems = testMatrix.getRow(u).indexList();			
			if (testItems != null) {
				DenseVector predictedForUser = getEstimation(u, testItems);			
				if (predictedForUser != null) {
					for (int i = 0; i < testItems.length; i++) {
						predicted.setValue(u, testItems[i], predictedForUser.getValue(i));
					}
				}
			}
		}
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}

	/**
	 * Estimate of ratings for a given user and a set of test items.
	 * 
	 * @param u The user number.
	 * @param testItemsRaw The list of items to be predicted.
	 * @return A list containing predicted rating scores.
	 */
	private DenseVector getEstimation(int u, int[] testItemsRaw) {
		SparseVector ytrainSp = rateMatrix.getRow(u);
		int[] trainItems = ytrainSp.indexList();
		int[] testItems = new int[testItemsRaw.length];

		if (trainItems != null) {
			DenseVector ytrain = ytrainSp.toDenseSubset(trainItems);

			for (int i = 0; i < trainItems.length; i++) {
				trainItems[i] = trainItems[i] - 1;
			}
			for (int i = 0; i < testItemsRaw.length; i++) {
				testItems[i] = testItemsRaw[i] - 1;
			}

			DenseMatrix Xtrain = itemFeatures.toDenseSubset(trainItems,index1toAll);
			DenseMatrix Xtest = itemFeatures.toDenseSubset(testItems,index1toAll);
			DenseMatrix KX_star = kernCompute(Xtrain, Xtest, false);  
			DenseMatrix K = kernCompute(Xtrain, Xtrain, true);
			DenseMatrix invK = K.inverse();	   	   
			DenseVector mu = KX_star.transpose().times(invK).times(ytrain);

			return mu;
		}
		else {
			DenseVector mu = new DenseVector(testItemsRaw.length);

			for (int i = 0; i < testItemsRaw.length; i++){
				double v = rateMatrix.getCol(testItemsRaw[i]).sum();

				if (v != 0){
					mu.setValue(i,rateMatrix.getCol(testItemsRaw[i]).average());
				}
				else{
					mu.setValue(i,(minValue+maxValue) / 2);
				}
			}

			return mu;
		}
	}

	/**
	 * Transform a vector by log function, exponential function or linear function.
	 * 
	 * @param x The input vector.
	 * @param transformType the type of function.
	 * @return The vector after transformation.
	 */
	private DenseVector expTransform(DenseVector x, int transformType){
		double limVal = 36;
		DenseVector y = new DenseVector(x.length());

		if (transformType == ATOX) {
			for (int i = 0; i < x.length(); i++) {
				double v = x.getValue(i);
				if (v < -limVal) {
					y.setValue(i, Math.exp(-limVal));
				}
				else if (v < limVal) {
					y.setValue(i, Math.exp(x.getValue(i)));
				}
				else {
					y.setValue(i, Math.exp(limVal));
				}
			}

			return y;
		}
		else if (transformType == XTOA) {
			for (int i = 0; i < x.length(); i++){
				y.setValue(i, Math.log(x.getValue(i)));
			}

			return y;
		}
		else if (transformType == GRADFACT) {
			return x;
		}
		else {
			System.out.println("Warning: no such transform type defined.");
			return x;
		}
	}

	/**
	 * Compute the squared Euclidean distance between the row vectors of one matrix and the row vectors of another matrix.
	 * Vectors in the two matrix have the same dimension. 
	 *
	 * @param X1 matrix1.
	 * @param X2 matrix2.
	 * @return The pairwise squared Euclidean distances.
	 */
	private DenseMatrix distancePairWise(DenseMatrix X1, DenseMatrix X2){
		int m = (X1.length())[0];
		int n = (X2.length())[0];
		DenseMatrix Z = new DenseMatrix(m,n);

		// maybe speed up by matrix operation x1^2+x2^2-x1*x2')
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double v = (X1.getRow(i).minus(X2.getRow(j))).power(2.0).sum();
				Z.setValue(i, j, Math.max(0,v));
			}
		}

		return Z;
	}

	/**
	 * Compute the squared euclidean distance between the row vectors of one matrix and a vector.
	 * Vectors in the two matrix have the same dimension. 
	 *
	 * @param X1 matrix1.
	 * @param x  vector.
	 * @return The pairwise squared euclidean distances.
	 */
	private DenseVector distancePairWise(DenseMatrix X1, DenseVector x){
		int m = (X1.length())[0];
		DenseVector z = new DenseVector(m);

		for (int i = 0; i < m; i++) {
			double v = X1.getRow(i).minus(x).power(2.0).sum();
			z.setValue(i, Math.max(0,v));
		}

		return z;
	}

	/**
	 * Compute RBF(radial basis function) kernel parameters for given distances.
	 *
	 * @param distance The distance matrix.
	 * @param kernInverseWidth The kernal inverse bandwidth.
	 * @param kernVarianceRbf The kern variance.
	 * @return The kernel matrix.
	 */
	private DenseMatrix rbfKernCompute(DenseMatrix distance, double  kernInverseWidth, double kernVarianceRbf){
		double wi2 = 0.5 * kernInverseWidth;
		DenseMatrix K = (distance.scale(0-wi2).exp(Math.E)).scale(kernVarianceRbf);

		return K;
	}

	/**
	 * Compute the gradient of RBF kernel with respect to input locations.
	 *
	 * @param x row locations against which gradients are being computed.	
	 * @param X2 column locations against which gradients are being computed. 
	 * @param kernInverseWidth The kernal inverse bandwidth.
	 * @param kernVarianceRbf The kern variance.
	 * @return the gradients.
	 */
	private DenseMatrix rbfKernGradXpoint(DenseVector x, DenseMatrix X2, double  kernInverseWidth, double kernVarianceRbf){
		// Gradient with respect to one point of x.
		int m = (X2.length())[0];
		int n = (X2.length())[1];
		DenseMatrix gX = new DenseMatrix(m,n);
		DenseVector dist = distancePairWise(X2, x);
		double wi2 = 0.5 * kernInverseWidth;
		DenseVector rbfPart = (dist.scale(-wi2).exp(Math.E)).scale(kernVarianceRbf);

		for(int i = 0; i < n; i++) {
			DenseVector temp = X2.getCol(i).sub(x.getValue(i));
			for (int j = 0 ;j < m; j++) {
				double v = kernInverseWidth * temp.getValue(j) * rbfPart.getValue(j);
				gX.setValue(j,i,v); 
			}
		}

		return gX;
	}

	/**
	 * Compute kernel parameters for vectors in matrix X1 and vectors in matrix X2.
	 *
	 * @param X1 the matrix1.
	 * @param X2 the matrix2.
	 * @param whiteNoiseFlag the flag for combining whiteNoise.
	 * @return The kernel matrix.
	 */
	private DenseMatrix kernCompute(DenseMatrix X1, DenseMatrix X2, boolean whiteNoiseFlag){
		double  kernInverseWidth = kernParam.getValue(0);
		double kernVarianceRbf = kernParam.getValue(1);
		double kernVarianceBias = kernParam.getValue(2);
		double kernVarianceWhite = kernParam.getValue(3);

		DenseMatrix distance = distancePairWise(X1,X2);
		DenseMatrix Krbf = rbfKernCompute(distance, kernInverseWidth, kernVarianceRbf);	
		DenseMatrix Kw = DenseMatrix.makeIdentity((Krbf.length())[0]).scale(kernVarianceWhite);

		if (whiteNoiseFlag) {
			return Krbf.plus(Kw).add(kernVarianceBias);
		}
		else {
			return Krbf.add(kernVarianceBias);
		}
	}

	/**
	 * Compute the gradients of the model (latent factors and kernel parameters) given one users ratings.
	 *
	 * @param user the userId.
	 * @param kernParam the kernel parameters.
	 * @param momentum the momentum.
	 * @param learningRate the learningRate.
	 * @return The kernel matrix.
	 */
	private DenseVector collabLogLikeGradients(int user, DenseVector kernParam, double momentum, double learningRate){
		double  kernInverseWidth = kernParam.getValue(0);
		double kernVarianceRbf = kernParam.getValue(1);
		double kernVarianceBias = kernParam.getValue(2);
		double kernVarianceWhite = kernParam.getValue(3);
		SparseVector mu = new SparseVector(itemCount); 
		double std = 1.0; 

		// g and g_param need to set to 0 every time:
		SparseMatrix g = new SparseMatrix(itemCount, featureCount);
		DenseVector g_param = new DenseVector(kernParam.length());
		SparseVector temp = new SparseVector(itemCount);
		int[] fullInd = rateMatrix.getRow(user).indexList();
		double[] valueList = rateMatrix.getRowRef(user).valueList();

		// Re-index items from 0 to itemCount-1:
		for (int i = 0; i < fullInd.length; i++) {
			fullInd[i] = fullInd[i]-1;
			temp.setValue(fullInd[i], valueList[i]);
		}
		temp = temp.minus(mu).scale(1.0/std);
		int len = fullInd.length;
		Sort.quickSort(fullInd, 0, len-1, true);
		int maxBlock = (int)(Math.ceil(len)/Math.ceil(len/1000.0));
		int t = (int) Math.ceil(len/maxBlock) + 1;
		int[] span = new int[t];

		for (int i = 0; i < t-1; i++) {
			span[i] = i*maxBlock;
		}

		span[t-1] = len;
		for (int block = 1; block < span.length; block++) {
			int l = span[block] - span[block - 1];
			int[] ind = new int[l];
			System.arraycopy(fullInd, span[block-1], ind, 0, l);
			DenseVector yprime = temp.toDenseSubset(ind);
			DenseMatrix X = itemFeatures.toDenseSubset(ind, index1toAll);
			int n = (X.length())[0];
			int q = (X.length())[1];

			// Compute kernel:
			DenseMatrix distance = distancePairWise(X,X);
			DenseMatrix Krbf = rbfKernCompute(distance, kernInverseWidth, kernVarianceRbf);	
			DenseMatrix Kw =  DenseMatrix.makeIdentity((Krbf.length())[0]).scale(kernVarianceWhite);           
			DenseMatrix K = Krbf.plus(Kw).add(kernVarianceBias);	   
			DenseMatrix invK = K.inverse();	
			DenseVector invKy = invK.times(yprime);
			DenseMatrix gK = invK.scale(-1.0).plus(invKy.outerProduct(invKy));               

			//Prepare to Compute Gradients with respect to X:
			DenseMatrix[] gKX = new DenseMatrix[n];

			for(int i = 0; i < n; i++){
				gKX[i] = rbfKernGradXpoint(X.getRow(i), X, kernInverseWidth, kernVarianceRbf).scale(2.0);
			}

			double dgKX = 0;//all zero matrix size(X)
			for (int i = 0; i < n; i++) {
				for(int j = 0; j < q; j++) { 
					gKX[i].setValue(i, j, dgKX);
				}
			}

			DenseMatrix gX = new DenseMatrix(n, q);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < q; j++) {
					double v = gX.getValue(i, j)+(gKX[i].getCol(j)).innerProduct(gK.getCol(i));
					gX.setValue(i, j, v);
				}
			}

			// assign gx in g subinds (in matlab simply g(ind, :) = gX)
			for (int i = 0; i < n; i++) {
				for ( int j = 0; j < q; j++) {
					g.setValue(ind[i], j, gX.getValue(i, j));
				}
			}

			// increase (in matlab g_param = g_param + kernGradient(kern, X, gK))
			double sum1 = 0; double sum2 = 0;
			for (int i = 0; i < n; i++){
				for (int j = 0; j < n; j++){
					double v = gK.getValue(i, j) * Krbf.getValue(i, j);
					sum1 += v;
					sum2 += v * distance.getValue(i,j);
				}
			}

			g_param.setValue(0, g_param.getValue(0)-kernParam.getValue(0) * 0.5 * sum2);
			g_param.setValue(1, g_param.getValue(1)+kernParam.getValue(1) * sum1 / kernVarianceRbf);
			g_param.setValue(2, g_param.getValue(2)+kernParam.getValue(2) * gK.sum());
			g_param.setValue(3, g_param.getValue(3)+kernParam.getValue(3) * (gK.diagonal()).sum());
		}

		itemFeaturesChange = (itemFeaturesChange.scale(momentum)).plus(g.scale(learningRate));
		itemFeatures = itemFeatures.plus(itemFeaturesChange);

		return g_param;
	}
}