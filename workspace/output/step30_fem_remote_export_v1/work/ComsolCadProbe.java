import com.comsol.model.*;
import com.comsol.model.util.*;

import java.util.Arrays;

public class ComsolCadProbe {
  private static final String INPUT =
      "/home/user/aegis_comsol_bucket_v1/input/full_runner_authoritative.x_t";
  public static Model run() throws Exception {
    Model model = ModelUtil.create("Model");
    model.label("AEGIS authoritative Pelton runner import probe");
    model.component().create("comp1", true);
    GeomSequence geom = model.component("comp1").geom().create("geom1", 3);
    geom.geomRep("cadps");
    geom.feature().create("imp1", "Import");
    geom.feature("imp1").set("filename", INPUT);
    geom.run("imp1");

    String[] names = geom.objectNames();
    System.out.println("COMSOL_CAD_PROBE_VERSION=comsol-cad-import-probe-v1");
    System.out.println("COMSOL_CAD_INPUT=" + INPUT);
    System.out.println("COMSOL_CAD_GEOM_REP=" + geom.geomRep());
    System.out.println("COMSOL_CAD_OBJECT_COUNT=" + names.length);
    System.out.println("COMSOL_CAD_GEOMETRY_TYPE=" + geom.getType());
    System.out.println("COMSOL_CAD_GEOMETRY_HAS_CAD_REP=" + geom.hasCadRep());
    System.out.println("COMSOL_CAD_GEOMETRY_DOMAINS=" + geom.getNDomains());
    System.out.println("COMSOL_CAD_GEOMETRY_BOUNDARIES=" + geom.getNBoundaries());
    System.out.println("COMSOL_CAD_GEOMETRY_EDGES=" + geom.getNEdges());
    System.out.println("COMSOL_CAD_GEOMETRY_VERTICES=" + geom.getNVertices());
    System.out.println("COMSOL_CAD_GEOMETRY_BOUNDS=" + Arrays.toString(geom.getBoundingBox()));
    for (String name : names) {
      GeomObject object = geom.obj(name);
      System.out.println(
          "COMSOL_CAD_OBJECT=" + name + "\t" + object.getType() + "\t"
              + object.hasCadRep() + "\t" + object.getNDomains() + "\t"
              + object.getNBoundaries() + "\t" + object.getNEdges() + "\t"
              + object.getNVertices() + "\t" + Arrays.toString(object.getBoundingBox()));
    }
    System.out.println("COMSOL_CAD_PROBE_OK objects=" + names.length);
    return model;
  }

  public static void main(String[] args) throws Exception {
    run();
  }
}
