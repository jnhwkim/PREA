package prea.util;

/**
 * This is a class implementing loss functions generally
 * used in collaborative filtering algorithms.
 * 
 * @author Joonseok Lee 
 * @author Mingxuan Sun 
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Loss {
	/**
	 * Asymmetric loss matrix/function.
	 * Note that this should be used only for discrete ratings.
	 * 
	 * @param realRate Target real rating.
	 * @param predictedRate Target predicted rating.
	 * @param minValue Possible minimum value of the domain.
	 * @param maxValue Possible maximum value of the domain.
	 * @return Asymmetric loss for the given target rating.
	 */
	/*
	 * Example of asymmetric loss matrix:
	 * 
	 *	{0, 0, 0, 7.5, 10, 12.5},
	 *	{0, 0, 0, 4, 6, 8},
	 *	{0, 0, 0, 1.5, 3, 4.5},
	 *	{3, 2, 1, 0, 0, 0},
	 *	{4, 3, 2, 0, 0, 0},
	 *	{5, 4, 3, 0, 0, 0}
	 */
	public static double asymmetricLoss(double realRate, double predictedRate, double minValue, double maxValue) {
		int real = bound(realRate, minValue, maxValue);
		int pred = bound(predictedRate, minValue, maxValue);
		int mid = (int) Math.ceil((double) maxValue / 2);
		double loss = 0;

		if (real <= mid && pred <= mid) {
			loss = 0;
		}
		else if (real > mid && pred <= mid) {
			loss = (double) real - (double) pred;
		}
		else if (real <= mid && pred > mid) {
			loss = (double) (pred - real) * (1 + (mid - real +1) * 0.5);
		}
		else {
			loss = 0;
		}
		return loss;    
	}
	
	/**
	 * Return the rounded index for a rating in a given range
	 * 
	 * @param value The target rating to bound.
	 * @param minValue The minimum of given range.
	 * @param maxValue The maximum of given range.
	 * @return the bounded index.
	 */
	private static int bound(double value, double minValue, double maxValue) {
		int v = (int) Math.round(value);
		return Math.min((int) maxValue, Math.max((int) minValue, v));
	}
}