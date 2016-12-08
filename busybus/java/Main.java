import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {
	
	final static int C = 3;
	final static int N = 6;
	
	static class Score implements Comparable<Score> {
		int constestant;
		double score;
		public Score(int contestant, double score) {
			this.constestant = contestant;
			this.score = score;
		}
		
		@Override
		public int compareTo(Score o) {
			double w = this.score - ((Score) o).score;
			if (w < 0) return -1;
			else if (w > 1) return 1;
			else return 0;
		}
	}
	
	public static void main(String[] args) {
		
		List<Strategy> strategies = new ArrayList<>();
		strategies.add(new AI_CLOCK(C, N));
		strategies.add(new AI_KCLOCK(C, N));
		strategies.add(new AI_GREEDY(C, N));
		
		int rank = strategies.size();
		
		int[] ranks = new int[rank];
		List<List<Double>> scores = new ArrayList<>();
		for (int i = 0; i < rank; i++)
			scores.add(new ArrayList<>());
		
		while (rank > 0) {
			System.out.println("Number of competitors: " + rank);
			
			World world = new World(C, N);
			
			// Strategy scores for this round
			// (nonnegative; max score loses)
			List<Score> K = new ArrayList<>();
			
			for (int n = 0; n < strategies.size(); ++n) {
				if (ranks[n] > 0) continue;
				
				Strategy strategy = strategies.get(n);
				
				System.out.println(" - Profiling " + strategy.name);
				
				Profiler report = new Profiler(world, strategy);
				double score = report.w;

				K.add(new Score(n, score));
				System.out.println("   *Score for this round: " + score);
			}
			assert(!K.isEmpty());
			
			Score maxScore = Collections.max(K);
			
			for (Score score : K) {
				int n = score.constestant;
				if (score.score == maxScore.score)
					ranks[n] = rank;
				List<Double> sc;
				if (scores.size() > 0)
					sc = new ArrayList<>(scores.get(n));
				else
					sc = new ArrayList<>();
				sc.add(score.score);
				scores.get(n).add(Utils.mean(sc));
			}
			rank = 0;
			for (int e : ranks)
				if (e == 0) rank += 1;
		}
		
		
		System.out.println("Final ranking:");
		for (rank = 1; rank <= ranks.length; ++rank) {
			System.out.println("Rank " + rank);
			for (int n = 0; n != ranks.length; ++n) {
				if (ranks[n] != rank) continue;
				System.out.println("   " + strategies.get(n).getName());
			}
		}
		
	}
	
}
