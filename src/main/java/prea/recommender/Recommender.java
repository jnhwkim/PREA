package prea.recommender;
import prea.data.structure.SparseMatrix;
import prea.util.EvaluationMetrics;

/**
 * Interface of general recommendation system.
 * Contains definition of learning and evaluating functions.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public interface Recommender {
	/**
	 * Interface of learning method.
	 * 
	 * @param rm A rating matrix with train data.
	 */
	public void buildModel(SparseMatrix rm);
	
	/**
	 * Interface of evaluation method.
	 * 
	 * @param tm A rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(SparseMatrix tm);
}
