import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class textparser {
    public static void main(String[] args) throws Exception
    {
        List ll = new ArrayList();

        File file = new File(
                "queryohsu1-63");

        // Note:  Double backquote is to avoid compiler
        // interpret words
        // like \test as \t (ie. as a escape sequence)

        // Creating an object of BufferedReader class
        BufferedReader br
                = new BufferedReader(new FileReader(file));

        // Declaring a string variable
        String st;
        // Condition holds true till
        // there is character in a string
        while ((st = br.readLine()) != null){
            // Print the string
//            Pattern pattern = Pattern.compile("\\<(.*?)\\>", Pattern.CASE_INSENSITIVE);
//            Matcher matcher = pattern.matcher(st);
//            boolean matchFound = matcher.find();
//            if(matchFound) {
//                System.out.println("Match found");
//                System.out.println(matcher.group(0));
//                String[] arrOfStr = st.split("\\<(.*?)\\>", -2);
//            } else {
//                System.out.println("Match not found");
//            }
//            System.out.println(st);
            String[] arrOfStr = st.split("\\<(.*?)\\>", -2);
            for (String a : arrOfStr) {
//                System.out.println(a);
                ll.add(a);
            }

        }

        System.out.println(ll);
        Iterator<String> it = ll.iterator();
        while(it.hasNext()){
            if (it.next().equals("")){
                it.remove();
            }
//            if (it.next().equals(" Description:")){
//                it.remove();
//            }
        }
        System.out.println(ll);
        Iterator<String> itf = ll.iterator();
        while(itf.hasNext()){
            if (itf.next().equals(" Description:")){
                itf.remove();
            }
        }
        System.out.println(ll);
        for (int i = 0; i < ll.size();i++)
        {
            if(((String) ll.get(i)).charAt(0) == ' '){
                ll.set(i, ((String) ll.get(i)).substring(1));
            }
//            ll.set(i, ((String) ll.get(i)).substring(1));
        }
//        String s = (String) ll.get(0);
//        StringBuilder sbStr = new StringBuilder(str);
        System.out.println(ll);
//        System.out.println(s.charAt(0));
//        System.out.println(((String) ll.get(0)).substring(1));


    }
}
