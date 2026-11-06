import com.comsol.model.*;
import com.comsol.model.util.*;

import java.util.Arrays;

public class ComsolBucketExport {
  private static final String INPUT =
      "/home/user/aegis_comsol_bucket_v1/input/full_runner_authoritative.x_t";
  private static final String OUTPUT_ROOT =
      "/home/user/aegis_comsol_bucket_v1/output/single_bucket_target_authoritative";
  private static final String BUCKET_OBJECT = "imp1(12)";

  public static Model run() throws Exception {
    Model model = ModelUtil.create("Model");
    model.label("AEGIS authoritative single Pelton bucket export");
    model.component().create("comp1", true);
    GeomSequence geom = model.component("comp1").geom().create("geom1", 3);
    geom.geomRep("cadps");
    geom.feature().create("imp1", "Import");
    geom.feature("imp1").set("filename", INPUT);
    geom.run("imp1");

    GeomObject bucket = geom.obj(BUCKET_OBJECT);
    if (!"solid".equals(bucket.getType()) || !bucket.hasCadRep() || bucket.getNDomains() != 1) {
      throw new IllegalStateException("Selected bucket is not one CAD solid");
    }
    if (bucket.getNBoundaries() != 30 || bucket.getNEdges() != 72 || bucket.getNVertices() != 44) {
      throw new IllegalStateException("Selected bucket topology differs from the 21 repeated bucket solids");
    }

    GeomMeasure measure = geom.measure();
    measure.selection().init(3);
    measure.selection().set(BUCKET_OBJECT, 1);
    double volume = measure.getVolume();
    double boundaryArea = measure.getBoundaryArea();
    double[] measuredBounds = measure.getBoundingBox();
    if (!(volume > 0.0) || !(boundaryArea > 0.0)) {
      throw new IllegalStateException("Selected bucket has non-positive volume or area");
    }

    GeomExport exporter = geom.export();
    exporter.selection().init();
    exporter.selection().set(BUCKET_OBJECT);

    exporter.setType("parasolidascii");
    exporter.setLengthUnit("m");
    exporter.setSplitInManifold(false);
    geom.export(OUTPUT_ROOT + ".x_t");

    exporter.setType("step");
    exporter.setLengthUnitSTEP("m");
    geom.export(OUTPUT_ROOT + ".step");

    exporter.setType("stlbin");
    geom.export(OUTPUT_ROOT + ".stl");

    System.out.println("COMSOL_BUCKET_EXPORT_VERSION=comsol-single-bucket-export-v1");
    System.out.println("COMSOL_BUCKET_SOURCE=" + INPUT);
    System.out.println("COMSOL_BUCKET_OBJECT=" + BUCKET_OBJECT);
    System.out.println("COMSOL_BUCKET_TYPE=" + bucket.getType());
    System.out.println("COMSOL_BUCKET_HAS_CAD_REP=" + bucket.hasCadRep());
    System.out.println("COMSOL_BUCKET_DOMAINS=" + bucket.getNDomains());
    System.out.println("COMSOL_BUCKET_BOUNDARIES=" + bucket.getNBoundaries());
    System.out.println("COMSOL_BUCKET_EDGES=" + bucket.getNEdges());
    System.out.println("COMSOL_BUCKET_VERTICES=" + bucket.getNVertices());
    System.out.println("COMSOL_BUCKET_BOUNDS=" + Arrays.toString(bucket.getBoundingBox()));
    System.out.println("COMSOL_BUCKET_MEASURED_BOUNDS=" + Arrays.toString(measuredBounds));
    System.out.println("COMSOL_BUCKET_VOLUME_M3=" + volume);
    System.out.println("COMSOL_BUCKET_BOUNDARY_AREA_M2=" + boundaryArea);
    System.out.println("COMSOL_BUCKET_PARASOLID=" + OUTPUT_ROOT + ".x_t");
    System.out.println("COMSOL_BUCKET_STEP=" + OUTPUT_ROOT + ".step");
    System.out.println("COMSOL_BUCKET_STL=" + OUTPUT_ROOT + ".stl");
    System.out.println("COMSOL_BUCKET_EXPORT_OK");
    return model;
  }

  public static void main(String[] args) throws Exception {
    run();
  }
}
