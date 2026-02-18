package net.irisshaders.iris.uniforms;

import net.minecraft.client.Minecraft;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.joml.Vector3d;

public class CapturedRenderingState {
	public static final CapturedRenderingState INSTANCE = new CapturedRenderingState();

	private static final Vector3d ZERO_VECTOR_3d = new Vector3d();

	private Matrix4fc gbufferModelView;
	private Matrix4fc gbufferProjection;
	private Vector3d fogColor;
	private float fogDensity;
	private float darknessLightFactor;
	private float tickDelta;
	private float realTickDelta;
	private int currentRenderedBlockEntity;

	private int currentRenderedEntity = -1;
	private int currentRenderedItem = -1;

	private float currentAlphaTest;
	private float cloudTime;

	private CapturedRenderingState() {
	}

	public Matrix4fc getGbufferModelView() {
		return gbufferModelView;
	}

	public void setGbufferModelView(Matrix4fc gbufferModelView) {
		this.gbufferModelView = gbufferModelView;
	}

	public Matrix4fc getGbufferProjection() {
		return gbufferProjection;
	}

	public void setGbufferProjection(Matrix4f gbufferProjection) {
		Matrix4f proj = new Matrix4f(gbufferProjection);
		// VulkanMod uses infinite far plane, producing Infinity in m00/m11.
		// Fix here so ALL consumers (MatrixUniforms, writeGbufferUniforms,
		// customUniforms) get finite values and projectionInverse doesn't have NaN.
		if (!Float.isFinite(proj.m00()) || !Float.isFinite(proj.m11())) {
			Minecraft mc = Minecraft.getInstance();
			double fovDegrees = 70.0;
			try {
				if (mc.gameRenderer != null) {
					fovDegrees = ((net.irisshaders.iris.mixin.GameRendererAccessor) mc.gameRenderer)
						.invokeGetFov(mc.gameRenderer.getMainCamera(),
							mc.getTimer().getGameTimeDeltaPartialTick(true), true);
				}
			} catch (Exception ignored) {}
			// Sanity check: valid Minecraft FOV is 30-170 degrees. Values outside
			// this range indicate the camera/renderer isn't ready yet (e.g., at HEAD
			// of renderLevel before camera setup). Fall back to 70 degrees.
			if (fovDegrees < 30.0 || fovDegrees > 170.0 || !Double.isFinite(fovDegrees)) {
				fovDegrees = 70.0;
			}
			float fovRad = (float)(fovDegrees * Math.PI / 180.0);
			float tanHalfFov = (float) Math.tan(fovRad / 2.0);
			var window = mc.getWindow();
			float aspect = (float) window.getWidth() / (float) window.getHeight();
			proj.m00(1.0f / (aspect * tanHalfFov));
			proj.m11(1.0f / tanHalfFov);
		}
		// Also fix infinite m22/m32 from infinite far plane
		if (proj.m23() != 0 && (!Float.isFinite(proj.m22()) || !Float.isFinite(proj.m32()))) {
			Minecraft mc = Minecraft.getInstance();
			float far = mc.gameRenderer != null ? mc.gameRenderer.getRenderDistance() : 256.0f;
			float near = 0.05f;
			proj.m22(-far / (far - near));
			proj.m32(-far * near / (far - near));
		}

		// Convert from VK [0,1] depth to GL [-1,1] depth convention.
		// Iris shader packs are written for OpenGL and reconstruct position via:
		//   vec4 viewPos = gbufferProjectionInverse * (screenPos * 2.0 - 1.0);
		// The "* 2.0 - 1.0" maps gl_FragCoord.z from [0,1] to [-1,1] (GL NDC).
		// gbufferProjectionInverse must be the inverse of a GL-convention projection
		// for the reconstruction to be correct. Without this, depth reconstruction
		// is wrong and the sky renders with a horizontal bar artifact.
		// Note: iris_ProjMat stays in VK depth (for correct vertex gl_Position).
		{
			float m22 = proj.m22();
			float m23 = proj.m23(); // -1 for perspective, 0 for ortho
			float m32 = proj.m32();
			float m33 = proj.m33(); // 0 for perspective, 1 for ortho
			if (Float.isFinite(m22) && Float.isFinite(m32)) {
				proj.m22(2.0f * m22 - m23);
				proj.m32(2.0f * m32 - m33);
			}
		}

		this.gbufferProjection = proj;
	}

	public Vector3d getFogColor() {
		if (Minecraft.getInstance().level == null || fogColor == null) {
			return ZERO_VECTOR_3d;
		}

		return fogColor;
	}

	public void setFogColor(float red, float green, float blue) {
		fogColor = new Vector3d(red, green, blue);
	}

	public float getFogDensity() {
		return fogDensity;
	}

	public void setFogDensity(float fogDensity) {
		this.fogDensity = fogDensity;
	}

	public float getTickDelta() {
		return tickDelta;
	}

	public void setTickDelta(float tickDelta) {
		this.tickDelta = tickDelta;
	}

	public float getRealTickDelta() {
		return realTickDelta;
	}

	public void setRealTickDelta(float tickDelta) {
		this.realTickDelta = tickDelta;
	}

	public void setCurrentBlockEntity(int entity) {
		this.currentRenderedBlockEntity = entity;
	}

	public int getCurrentRenderedBlockEntity() {
		return currentRenderedBlockEntity;
	}

	public void setCurrentEntity(int entity) {
		this.currentRenderedEntity = entity;
	}

	public int getCurrentRenderedEntity() {
		return currentRenderedEntity;
	}

	public int getCurrentRenderedItem() {
		return currentRenderedItem;
	}

	public void setCurrentRenderedItem(int item) {
		this.currentRenderedItem = item;
	}

	public float getCurrentAlphaTest() {
		return currentAlphaTest;
	}

	public void setCurrentAlphaTest(float alphaTest) {
		this.currentAlphaTest = alphaTest;
	}

	public float getDarknessLightFactor() {
		return darknessLightFactor;
	}

	public void setDarknessLightFactor(float factor) {
		darknessLightFactor = factor;
	}

	public float getCloudTime() {
		return this.cloudTime;
	}

	public void setCloudTime(float cloudTime) {
		this.cloudTime = cloudTime;
	}
}
