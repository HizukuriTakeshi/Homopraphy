import org.opencv.core.Core;

public class HelloOpenCV {

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		new FindObject().run("./resources/f.JPG","./resources/e.JPG","./resources/result.jpg");
	}
}