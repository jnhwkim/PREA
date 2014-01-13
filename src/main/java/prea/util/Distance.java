package prea.util;

/**
 * This is a class implementing various distance measures of two vectors.
 * 
 * @author Mingxuan Sun 
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Distance {
	/**
	 * Return NDCG score for a ranked list given the scores and relevance of items in the list.
	 * @param uItemID array of userIds
	 * @param relevance array of relevances for userIds
	 * @param vItemID array of userIds may/maynot be in the same order as uItemID
	 * @param userScore array of scores for userIds
	 * @return the NDCG score
	 */
	public static double distanceNDCG(int[] uItemID, double[] relevance, int[] vItemID, double[] userScore) {
		//re-order so that relevance and scores corresponds to the same user
		int m = relevance.length;
		Sort.quickSort(uItemID, relevance, 0, m - 1, true);
		Sort.quickSort(vItemID, userScore, 0, m - 1, true);
		//now compute ndcg
		double dcg = 0.0;
		double idcg = 0.0;
		int[] id = new int[m];

		for (int i = 0; i < m; i++){
			id[i] = i;
		}

		Sort.quickSort(userScore, id, 0, m - 1, false);

		for (int i = 0; i < m; i++){
			double s = Math.pow(2.0, relevance[id[i]]);
			double t = Math.log(1.0+(double)(i+1));
			t = t / Math.log(2.0);
			dcg += (s-1.0)/t;
		}

		Sort.quickSort(relevance, id, 0, m - 1, false);

		for (int i = 0; i < m; i++){
			double s = Math.pow(2.0, relevance[i]);
			double t = Math.log(1.0+(double)(i+1));
			t = t / Math.log(2.0);
			idcg += (s-1.0)/t;
		}

		return dcg / idcg;
	}
	/**
	 * Given two permutation, re-index the id to the set {1,2,.., n}, where n is the total items of the two permutations.
	 * The original itemId will be re-ordered.
	 * @param uItemID array of itemID sorted increasingly 	
	 * @param vItemID array of itemID sorted increasingly
	 */
	private static void changePermutationIndex(int[] uItemID, int[] vItemID){
		int i = 0;int j = 0;int k = 0;
		int currID1, currID2;	
		while (true) {
			if (i >= uItemID.length || j >= vItemID.length) break;		
			currID1 = uItemID[i];
			currID2 = vItemID[j];			
			if (currID1 < currID2) {
				uItemID[i] = k+1;
				k++;
				i++;
			}
			else if (currID1 > currID2) {
				vItemID[j] = k+1;
				k++;
				j++;
			} 
			else { // when the two items are equal
				uItemID[i] = k+1;
				vItemID[j] = k+1;
				k++;
				i++;j++;
			}
		}
		if(i >= uItemID.length) {
			for ( ; j < vItemID.length; j++){
				vItemID[j] = k+1;
				k++;
			}
		}
		else {
			for ( ; i < uItemID.length; i++) {
				uItemID[i] = k+1;
				k++;
			}
		}  
	}
	/**
	 * Return intermediate Kendall's Tau distance for two rankings parsed by prb
	 * 
	 * @param uItemID array of itemID sorted increasingly
	 * @param uPrb array of probability for user u
	 * @param vItemID array of itemID for user v sorted increasingly
	 * @param vPrb array of probability for user v
	 * @param n the number of total items
	 * @return the Kendall's Tau scores
	 */
	private static double distanceKendallParsed(int[] uItemID, double[] uPrb, int[] vItemID, double[] vPrb, int n) {
		if (n > 1) {
			int i = 0;
			int j = 0;
			int k = 0;
			int currID1, currID2;
			int[] globalIndex = new int[uItemID.length+vItemID.length];
			int[] globalRate1 = new int[uItemID.length+vItemID.length];
			int[] globalRate2 = new int[uItemID.length+vItemID.length];

			while (true) {
				if (i >= uItemID.length || j >= vItemID.length) break;

				currID1 = uItemID[i];
				currID2 = vItemID[j];

				if (currID1 < currID2) {
					globalIndex[k] = currID1;
					globalRate1[k] = i+1;
					globalRate2[k] = 0;
					k++;
					i++;
				}
				else if (currID1 > currID2) {
					globalIndex[k] = currID2;
					globalRate2[k] = j+1;
					globalRate1[k] = 0;
					k++;
					j++;
				} 
				else { // when the two items are equal
					globalIndex[k] = currID1;
					globalRate1[k] = i + 1;
					globalRate2[k] = j + 1;
					k++;
					i++;
					j++;
				}
			}

			if(i >= uItemID.length) {
				for ( ; j < vItemID.length; j++){
					globalIndex[k] = vItemID[j];
					globalRate2[k] = j + 1;
					globalRate1[k] = 0;
					k++;
				}
			}
			else {
				for ( ; i < uItemID.length; i++) {
					globalIndex[k] = uItemID[i];
					globalRate1[k] = i + 1;
					globalRate2[k] = 0;
					k++;
				}
			}  

			// Now, compute the Kendall:
			int num1 = 0, num3 = 0;
			double sum1 = 0, sum2 = 0, sum3 = 0, sum = 0;
			double prbi1 = 0, prbi2 = 0, prbj1 = 0, prbj2 = 0;
			double a1_ij = 0, a2_ij = 0;

			for(i = 0; i < k; i++){
				// to compute all a(j',i'),item index j'<i' and j' not in any
				num1 = globalIndex[i]-1-i;
				prbi1 = 0.0; prbi2 = 0.0;

				if (globalRate1[i] > 0) {
					prbi1 = uPrb[globalRate1[i]-1];
				}

				if (globalRate2[i] > 0) {
					prbi2 = vPrb[globalRate2[i]-1];
				}

				if (globalRate1[i] == 0 || globalRate2[i] == 0){
					sum1 = 0.0;
				}
				else{
					sum1 = num1 * (2 * prbi1 - 1) * (2 * prbi2 - 1);
				}   

				// compute special a(i',j'):
				sum2 = 0.0;
				for (j = i + 1 ;j < k; j++) {
					prbj1 = 0.0; prbj2 = 0.0;
					if(globalRate1[j] > 0){
						prbj1 = uPrb[globalRate1[j]-1];
					}
					if(globalRate2[j] > 0){		   
						prbj2 = vPrb[globalRate2[j]-1];
					}
					a1_ij = preferProbability(prbi1,prbj1);
					a2_ij = preferProbability(prbi2,prbj2);
					sum2 = sum2 + a1_ij * a2_ij; 
				}

				// compute all the rest a(i',j'),j'>i' and j' not in any:
				num3 = n - globalIndex[i] - (k - i - 1);

				if(globalRate1[i] == 0 || globalRate2[i] == 0){
					sum3 = 0.0;
				}
				else{ 
					sum3 = num3 * (2 * (1 - prbi1) - 1) * (2 * (1 - prbi2) - 1);
				} 

				sum = sum + (sum1 + sum2 + sum3);     
			}

			return 0.5 - sum / ((double) n * (n-1));
		}
		else {
			return 0;
		}
	}

	/**
	 * Return the Kendall's Tau distance for two rankings.
	 * 
	 * @param uItemID array of itemID 
	 * @param uScore array of score for user u
	 * @param vItemID array of itemID for user v
	 * @param vScore array of scores for user v
	 * @param n the number of total items
	 * @return the Kendall's Tau distance
	 */
	public static double distanceKendall(int[] uItemID, double[] uScore, int[] vItemID, double[] vScore, int n) {
		Sort.quickSort(uItemID, uScore, 0, uItemID.length - 1, true);		
		Sort.quickSort(vItemID, vScore, 0, vItemID.length - 1, true);		
		changePermutationIndex(uItemID, vItemID);
		double[] uPrb = new double[uItemID.length];
		computeAverageRank(uScore, uPrb);
		double[] vPrb = new double[vItemID.length];
		computeAverageRank(vScore, vPrb);		
		return distanceKendallParsed(uItemID, uPrb, vItemID, vPrb, n);
	}

	/**
	 * Return the Spearman distance for two rankings parsed by probability
	 * 
	 * @param uItemID array of itemID sorted increasingly
	 * @param uPrb array of probability for user u
	 * @param vItemID array of itemID for user v sorted increasingly
	 * @param vPrb array of probability for user v
	 * @param n the number of total items
	 * @return the Spearman scores
	 */
	public static double distanceSpearmanParsed(int[] uItemID, double[] uPrb, int[] vItemID, double[] vPrb, int n){
		if(n > 1){
			int i = 0;
			int j = 0;
			int k = 0;
			int currID1, currID2;
			double temp = 0;

			double dotProd = 0.0;
			double defaultPrbu = 0.5;
			double defaultPrbv = 0.5;

			while (true) {
				if (i >= uItemID.length || j >= vItemID.length) break;

				currID1 = uItemID[i];
				currID2 = vItemID[j];

				if (currID1 < currID2) {
					dotProd += uPrb[i] * defaultPrbv;
					i++;
					k++;
				} 
				else if (currID1 > currID2) {
					dotProd += vPrb[j] * defaultPrbu;
					j++;
					k++;
				} 
				else { // When the two items are equal
					dotProd += uPrb[i] * vPrb[j];
					i++;j++;
					k++;
				}
			}

			if(i >= uItemID.length){
				temp = 0;
				for(; j < vItemID.length; j++){ 
					temp += vPrb[j]; 
					k++;
				}
				dotProd += temp * defaultPrbu;      
			}
			else {
				temp = 0;
				for(; i < uItemID.length; i++){
					temp += uPrb[i];
					k++;
				}
				dotProd += temp * defaultPrbv;
			}

			dotProd += (n - k) * defaultPrbu * defaultPrbv;

			return (double)(2 * n + 1) / (double)(n-1) - dotProd * 6.0 * (double)(n+1) / (double)(n * (n - 1));
		}
		else{
			return 0;
		}
	}

	/**
	 * Return Spearman distance for two rankings.
	 * 
	 * @param uItemID array of itemID 
	 * @param uScore array of score for user u
	 * @param vItemID array of itemID for user v 
	 * @param vScore array of scores for user v
	 * @param n the number of total items
	 * @return the Spearman distance
	 */
	public static double distanceSpearman(int[] uItemID, double[] uScore, int[] vItemID, double[] vScore, int n) {
		Sort.quickSort(uItemID, uScore, 0, uItemID.length - 1, true);
		Sort.quickSort(vItemID, vScore, 0, vItemID.length - 1, true);
		changePermutationIndex(uItemID,vItemID);
		double[] uPrb = new double[uItemID.length];
		computeAverageRank(uScore, uPrb);
		double[] vPrb = new double[vItemID.length];
		computeAverageRank(vScore, vPrb);
		return distanceSpearmanParsed(uItemID, uPrb, vItemID, vPrb, n);
	}

	/**
	 * Return the average rank of each score with/without ties prb=(lowrank+(tie-1)/2)/(k+1)
	 * 
	 * @param score The array of scores 
	 * @param prb The array of average ranks will be filled.
	 */
	public static void computeAverageRank(double[] score, double[] prb) {
		int k = score.length;
		int[] id = new int[k];
		for (int i = 0; i < k; i++){
			id[i] = i;
		}
		double[] scoreClone = (double[])score.clone();
		Sort.quickSort(scoreClone,id,0,k-1,false); 
		int[] lowRank = new int[k];
		int[] tie = new int[k];
		int j = 1;
		lowRank[0]= 1;
		tie[0] = 1;
		for (int i = 1; i < k; i++){
			if(scoreClone[i] == scoreClone[i-1]){
				tie[j-1] += 1;
			}
			else{
				lowRank[j] = lowRank[j-1] + tie[j-1];
				tie[j] = 1;
				j += 1;
			}
		}
		for(int i = 0; i < j; i++){
			for(int l = lowRank[i]; l < lowRank[i] + tie[i]; l++){
				prb[id[l - 1]] = ((double) lowRank[i] + (double) (tie[i] - 1.0) / 2.0) / (double)(k + 1);
			}
		}
	}

	/**
	 * Return the probability of the item i is preferred to the item j.
	 * 
	 * @param avgRank_i The average rank of item i
	 * @param avgRank_j The average rank of item j
	 * @return the intermediate Kendall scores
	 */
	private static double preferProbability(double avgRank_i, double avgRank_j){
		double a_ij = 0;

		if (avgRank_i > 0.0 && avgRank_j > 0.0) {
			a_ij = Math.signum(avgRank_j - avgRank_i);
		}
		else if (avgRank_i > 0.0) { 
			a_ij = 2 * (1 - avgRank_i) - 1;
		}
		else if (avgRank_j > 0.0) {
			a_ij = 2 * avgRank_j - 1;
		}
		else {
			a_ij = 0.0;
		}

		return a_ij;
	}
}
