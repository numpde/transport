import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class Utils {
	
	public static double mean(List<? extends Number> list) {
		assert list.size() != 0 : "empty list";
		
		long l = 0;
		for (Number e : list)
			l += e.longValue();
		return l / (double) list.size();

	}
	
	public static int randInt(int min, int max) {
		return ThreadLocalRandom.current().nextInt(min, max + 1);
	}
	
}
