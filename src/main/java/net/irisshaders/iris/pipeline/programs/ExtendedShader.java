package net.irisshaders.iris.pipeline.programs;

import com.mojang.blaze3d.preprocessor.GlslPreprocessor;
import com.mojang.blaze3d.shaders.Program;
import com.mojang.blaze3d.shaders.ProgramManager;
import com.mojang.blaze3d.shaders.Uniform;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.VertexFormat;
import net.irisshaders.iris.Iris;
import net.irisshaders.iris.gl.GLDebug;
import net.irisshaders.iris.gl.IrisRenderSystem;
import net.irisshaders.iris.gl.blending.AlphaTest;
import net.irisshaders.iris.gl.blending.BlendModeOverride;
import net.irisshaders.iris.gl.blending.BufferBlendOverride;
import net.irisshaders.iris.gl.framebuffer.GlFramebuffer;
import net.irisshaders.iris.gl.image.ImageHolder;
import net.irisshaders.iris.gl.program.IrisProgramTypes;
import net.irisshaders.iris.gl.program.ProgramImages;
import net.irisshaders.iris.gl.program.ProgramSamplers;
import net.irisshaders.iris.gl.program.ProgramUniforms;
import net.irisshaders.iris.gl.sampler.SamplerHolder;
import net.irisshaders.iris.gl.texture.TextureType;
import net.irisshaders.iris.gl.uniform.DynamicLocationalUniformHolder;
import net.irisshaders.iris.mixinterface.ShaderInstanceInterface;
import net.irisshaders.iris.pipeline.IrisRenderingPipeline;
import net.irisshaders.iris.samplers.IrisSamplers;
import net.irisshaders.iris.shadows.ShadowRenderer;
import net.irisshaders.iris.uniforms.CapturedRenderingState;
import net.irisshaders.iris.uniforms.custom.CustomUniforms;
import net.irisshaders.iris.vertices.ImmediateState;
import net.irisshaders.iris.gl.shader.ShaderType;
import net.irisshaders.iris.vulkan.shader.IrisSPIRVCompiler;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.ShaderInstance;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.Resource;
import net.minecraft.server.packs.resources.ResourceProvider;
import net.vulkanmod.interfaces.ShaderMixed;
import net.irisshaders.iris.gl.uniform.IrisUniformBuffer;
import net.vulkanmod.vulkan.VRenderSystem;
import net.vulkanmod.vulkan.shader.GraphicsPipeline;
import net.vulkanmod.vulkan.shader.Pipeline;
import net.vulkanmod.vulkan.shader.SPIRVUtils;
import net.vulkanmod.vulkan.shader.descriptor.ImageDescriptor;
import net.vulkanmod.vulkan.shader.descriptor.ManualUBO;
import net.vulkanmod.vulkan.shader.descriptor.UBO;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.joml.Matrix3f;
import org.joml.Matrix4f;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ExtendedShader extends ShaderInstance implements ShaderInstanceInterface {
	// GL constants (inlined from KHRDebug/ARBTextureSwizzle/GL30C)
	private static final int GL_PROGRAM = 0x82E2;
	private static final int GL_SHADER = 0x82E1;
	private static final int GL_TEXTURE_SWIZZLE_RGBA = 0x8E46;
	private static final int GL_RED = 0x1903;

	private static final Matrix4f identity;
	private static ExtendedShader lastApplied;

	static {
		identity = new Matrix4f();
		identity.identity();
	}

	private final boolean intensitySwizzle;
	private final List<BufferBlendOverride> bufferBlendOverrides;
	private final boolean hasOverrides;
	private final Uniform modelViewInverse;
	private final Uniform projectionInverse;
	private final Uniform normalMatrix;
	private final CustomUniforms customUniforms;
	private final IrisRenderingPipeline parent;
	private final ProgramUniforms uniforms;
	private final ProgramSamplers samplers;
	private final ProgramImages images;
	private final GlFramebuffer writingToBeforeTranslucent;
	private final GlFramebuffer writingToAfterTranslucent;
	private final BlendModeOverride blendModeOverride;
	private final float alphaTest;
	private final boolean usesTessellation;
	private final Matrix4f tempMatrix4f = new Matrix4f();
	private final Matrix3f tempMatrix3f = new Matrix3f();
	private final float[] tempFloats = new float[16];
	private final float[] tempFloats2 = new float[9];
	private Program geometry, tessControl, tessEval;

	// Phase 7: Vulkan uniform buffer for this shader
	private IrisUniformBuffer irisUniformBuffer;
	private ManualUBO irisManualUBO;
	private int irisProgramId;
	private List<IrisSPIRVCompiler.UniformField> sharedUniforms;

	// Keep iris_ProjMat in Vulkan [0,1] depth, matching VulkanMod's built-in MVP behavior.
	// The depth patch (GL→VK conversion in vertex shader) is skipped when true.
	private static final boolean DIAG_VK_DEPTH_BYPASS = true;

	// When true, apply() skips framebuffer binding, blend mode, and pipeline binding.
	// Used by updateUniformsOnly() to populate the UBO without render pass side effects.
	private boolean uniformsOnlyMode = false;

	public ExtendedShader(ResourceProvider resourceFactory, String string, VertexFormat vertexFormat, boolean usesTessellation,
						  GlFramebuffer writingToBeforeTranslucent, GlFramebuffer writingToAfterTranslucent,
						  BlendModeOverride blendModeOverride, AlphaTest alphaTest,
						  Consumer<DynamicLocationalUniformHolder> uniformCreator, BiConsumer<SamplerHolder, ImageHolder> samplerCreator, boolean isIntensity,
						  IrisRenderingPipeline parent, @Nullable List<BufferBlendOverride> bufferBlendOverrides, CustomUniforms customUniforms) throws IOException {
		super(resourceFactory, string, vertexFormat);

		// VulkanMod nulls out GL programs - guard debug naming calls
		if (this.getVertexProgram() != null) {
			GLDebug.nameObject(GL_SHADER, this.getVertexProgram().getId(), string + "_vertex.vsh");
		}
		if (this.getFragmentProgram() != null) {
			GLDebug.nameObject(GL_SHADER, this.getFragmentProgram().getId(), string + "_fragment.fsh");
		}

		int programId = this.getId();

		if (programId != 0) {
			GLDebug.nameObject(GL_PROGRAM, programId, string);
		}

		// Phase 7: Read shader sources early to determine UBO layout BEFORE ProgramUniforms
		// so that getUniformLocation() returns real byte offsets instead of -1
		this.irisProgramId = IrisRenderSystem.allocateIrisProgramId();
		String vshSrc = readShaderSource(resourceFactory, ".vsh");
		String fshSrc = readShaderSource(resourceFactory, ".fsh");
		if (vshSrc != null && fshSrc != null) {
			// Collect non-opaque uniforms from both shaders and merge into one list
			@SuppressWarnings("unchecked")
			List<IrisSPIRVCompiler.UniformField> merged = IrisSPIRVCompiler.mergeUniforms(
				IrisSPIRVCompiler.collectLooseUniforms(vshSrc),
				IrisSPIRVCompiler.collectLooseUniforms(fshSrc)
			);
			this.sharedUniforms = merged;

			// Preprocess one shader with the shared UBO to parse the std140 layout
			String sampleVulkan = IrisSPIRVCompiler.prepareForVulkan(fshSrc, merged);
			this.irisUniformBuffer = IrisUniformBuffer.fromVulkanGLSL(sampleVulkan);
			// NOTE: transposeMatrices is intentionally FALSE. Composite/final passes
			// (Program class) work correctly with column-major data, proving SPIR-V
			// uses ColMajor (default). Transposing here makes entities invisible.
			// The pane artifacts have a different root cause (likely UBO offset mismatch).

			// Register so getUniformLocation(irisProgramId, name) returns byte offsets
			IrisRenderSystem.registerUniformBuffer(this.irisProgramId, this.irisUniformBuffer);
			// Register sampler names so getUniformLocation returns non-(-1) for samplers
			java.util.Set<String> samplerNames = new java.util.HashSet<>();
			collectSamplerNamesFromSource(vshSrc, samplerNames);
			collectSamplerNamesFromSource(fshSrc, samplerNames);
			IrisRenderSystem.registerSamplerNames(this.irisProgramId, samplerNames);

			Iris.logger.debug("Registered IrisUniformBuffer for {} with {} fields, {} bytes, {} samplers",
				string, irisUniformBuffer.getFields().size(), irisUniformBuffer.getUsedSize(), samplerNames.size());
		}

		// Use irisProgramId for BOTH uniform and sampler location lookups
		ProgramUniforms.Builder uniformBuilder = ProgramUniforms.builder(string, this.irisProgramId);
		ProgramSamplers.Builder samplerBuilder = ProgramSamplers.builder(this.irisProgramId, IrisSamplers.WORLD_RESERVED_TEXTURE_UNITS);
		uniformCreator.accept(uniformBuilder);
		ProgramImages.Builder builder = ProgramImages.builder(this.irisProgramId);
		samplerCreator.accept(samplerBuilder, builder);
		customUniforms.mapholderToPass(uniformBuilder, this);
		this.usesTessellation = usesTessellation;

		uniforms = uniformBuilder.buildUniforms();
		this.customUniforms = customUniforms;

		// Get sampler name → texture unit mapping BEFORE building ProgramSamplers
		// This mapping is used by createVulkanPipeline to set correct imageIdx on ImageDescriptors
		java.util.Map<String, Integer> samplerUnitMap = samplerBuilder.getSamplerNameToUnit();

		samplers = samplerBuilder.build();
		images = builder.build();
		this.writingToBeforeTranslucent = writingToBeforeTranslucent;
		this.writingToAfterTranslucent = writingToAfterTranslucent;
		this.blendModeOverride = blendModeOverride;
		this.bufferBlendOverrides = bufferBlendOverrides;
		this.hasOverrides = bufferBlendOverrides != null && !bufferBlendOverrides.isEmpty();
		this.alphaTest = alphaTest.reference();
		this.parent = parent;

		this.modelViewInverse = this.getUniform("ModelViewMatInverse");
		this.projectionInverse = this.getUniform("ProjMatInverse");
		this.normalMatrix = this.getUniform("NormalMat");

		this.intensitySwizzle = isIntensity;

		// Create VulkanMod GraphicsPipeline AFTER ProgramSamplers is built
		// so we can use the actual sampler→textureUnit mapping for imageIdx
		createVulkanPipeline(resourceFactory, string, vertexFormat, samplerUnitMap);
	}

	public boolean isIntensitySwizzle() {
		return intensitySwizzle;
	}

	@Override
	public void clear() {
		ProgramUniforms.clearActiveUniforms();
		ProgramSamplers.clearActiveSamplers();
		lastApplied = null;

		if (this.blendModeOverride != null || hasOverrides) {
			BlendModeOverride.restore();
		}

		// Do NOT bind main render target here.
		// In OpenGL Iris, bindWrite(false) was cheap (just glBindFramebuffer).
		// In Vulkan, it causes a render pass transition — ending the gbuffer pass
		// and starting a new one on the main framebuffer. This happens after EVERY
		// entity RenderType batch, creating a gbuffer→main→gbuffer ping-pong that
		// costs 2 render pass transitions per batch.
		// The gbuffer render pass stays active for subsequent entity/hand draws.
		// Phase transitions (beginTranslucents, finalizeLevelRendering) bind their
		// own framebuffers, so the main FB doesn't need to be restored per-draw.
	}

	private static final org.joml.Matrix4f IDENTITY_MATRIX = new org.joml.Matrix4f();

	@Override
	public void apply() {
		// Update MC uniform fields from RenderSystem.
		// ShaderInstanceM.apply() returns early for non-legacy shaders (isLegacy=false),
		// and ExtendedShader doesn't call super.apply(), so we must do this ourselves.
		if (MODEL_VIEW_MATRIX != null) {
			// Entity vertices are in world-relative coordinates (translated from camera
			// but NOT rotated by camera). RenderSystem.getModelViewMatrix() provides the
			// correct transform for each render phase:
			//   - Entities: camera rotation Rx(pitch) * Ry(yaw+180)
			//   - Hand: identity (hand vertices are pre-transformed to view space)
			//   - Sky: identity (sky vertices are pre-transformed)
			//   - Shadow: shadow camera matrix (set by Iris shadow renderer)
			MODEL_VIEW_MATRIX.set(RenderSystem.getModelViewMatrix());
		}
		if (PROJECTION_MATRIX != null) {
			PROJECTION_MATRIX.set(RenderSystem.getProjectionMatrix());
			java.nio.FloatBuffer projBuf = PROJECTION_MATRIX.getFloatBuffer();

			// VulkanMod's getDepthFar() returns POSITIVE_INFINITY, which can cause
			// m00/m11 to be Infinity in the projection matrix. Entity vertex shaders
			// use iris_ProjMat directly in ftransform(), so Infinity values project
			// all vertices to extreme positions, creating line artifacts.
			// Rebuild the projection from actual FOV/aspect/far when infinite.
			float m00 = projBuf.get(0);  // column-major: index 0 = m00
			float m11 = projBuf.get(5);  // column-major: index 5 = m11
			if (!Float.isFinite(m00) || !Float.isFinite(m11)) {
				Minecraft mcClient = Minecraft.getInstance();
				double fovDegrees = 70.0;
				try {
					if (mcClient.gameRenderer != null) {
						fovDegrees = ((net.irisshaders.iris.mixin.GameRendererAccessor) mcClient.gameRenderer)
							.invokeGetFov(mcClient.gameRenderer.getMainCamera(),
								mcClient.getTimer().getGameTimeDeltaPartialTick(true), true);
					}
				} catch (Exception ignored) {}
				if (fovDegrees < 1.0 || !Double.isFinite(fovDegrees)) fovDegrees = 70.0;
				float fovRad = (float)(fovDegrees * Math.PI / 180.0);
				float tanHalfFov = (float) Math.tan(fovRad / 2.0);
				var window = mcClient.getWindow();
				float aspect = (float) window.getWidth() / (float) window.getHeight();
				float newM00 = 1.0f / (aspect * tanHalfFov);
				float newM11 = 1.0f / tanHalfFov;
				projBuf.put(0, newM00);
				projBuf.put(5, newM11);
				m11 = newM11;

				// Also rebuild m22/m32 with finite far for depth consistency with terrain
				float far = mcClient.gameRenderer != null ? mcClient.gameRenderer.getRenderDistance() : 256.0f;
				float near = 0.05f;
				// Vulkan [0,1] depth: m22 = -far/(far-near), m32 = -far*near/(far-near)
				projBuf.put(10, -far / (far - near));       // m22 at index 10
				projBuf.put(14, -far * near / (far - near)); // m32 at index 14
			}

			// Also fix NaN/infinite m22/m32 even when m00/m11 are finite.
			// This happens when MC manually constructs perspective with infinite far
			// (useShaderTransparency() path: -(∞+near)/(∞-near) = NaN from ∞/∞).
			// Without valid m22/m32, near-plane clipping is disabled (NaN comparisons
			// always return false), causing vertices at z≈0 to project to extreme positions.
			{
				float m22 = projBuf.get(10);
				float m32 = projBuf.get(14);
				float m23 = projBuf.get(11); // column-major index 11 = m23
				if (m23 != 0 && (!Float.isFinite(m22) || !Float.isFinite(m32))) {
					Minecraft mcClient2 = Minecraft.getInstance();
					float far2 = mcClient2.gameRenderer != null ? mcClient2.gameRenderer.getRenderDistance() : 256.0f;
					float near2 = 0.05f;
					projBuf.put(10, -far2 / (far2 - near2));
					projBuf.put(14, -far2 * near2 / (far2 - near2));
				}
			}

			// VulkanMod handles Y-flip purely through negative viewport height
			// (Framebuffer.beginRenderPass → setViewport → viewport.height = -height).
			// VulkanMod does NOT negate m11 for terrain, so we must NOT negate it
			// here either — doing so would cause a double Y-flip (entities upside-down).

			// Convert iris_ProjMat from Vulkan [0,1] depth range to GL [-1,1] depth range.
			// VulkanMod's Matrix4fM mixin forces all projection matrices to use zZeroToOne=true,
			// but Iris shader packs are written for OpenGL and expect [-1,1] depth range in
			// iris_ProjMat/gl_ProjectionMatrix. Without this conversion, depth calculations in
			// the vertex shader (ftransform) produce wrong Z values, causing depth test failures
			// and incorrect fragment depths in the gbuffer.
			//
			// SKIP for shadow passes: ShadowMatrices.createOrthoMatrix()/createPerspectiveMatrix()
			// construct matrices manually with GL [-1,1] conventions (not through JOML's ortho()/
			// perspective() which VulkanMod intercepts to force zZeroToOne=true). Converting an
			// already-GL matrix would double-convert, corrupting shadow depth.
			if (!ShadowRenderer.ACTIVE && !DIAG_VK_DEPTH_BYPASS) {
				// General formula (works for both perspective and orthographic):
				//   M[2][2]_gl = 2 * M[2][2]_vk - M[3][2]_vk
				//   M[2][3]_gl = 2 * M[2][3]_vk - M[3][3]_vk
				float col2row2 = projBuf.get(10); // M[2][2] - depth scale
				float col2row3 = projBuf.get(11); // M[3][2] - perspective divide (-1 for persp, 0 for ortho)
				float col3row2 = projBuf.get(14); // M[2][3] - depth offset
				float col3row3 = projBuf.get(15); // M[3][3] - (0 for persp, 1 for ortho)
				projBuf.put(10, 2.0f * col2row2 - col2row3);
				projBuf.put(14, 2.0f * col3row2 - col3row3);
			}
		}
		if (TEXTURE_MATRIX != null) {
			TEXTURE_MATRIX.set(RenderSystem.getTextureMatrix());
		}
		if (COLOR_MODULATOR != null) {
			COLOR_MODULATOR.set(RenderSystem.getShaderColor());
		}
		if (GLINT_ALPHA != null) {
			GLINT_ALPHA.set(RenderSystem.getShaderGlintAlpha());
		}
		if (FOG_START != null) {
			FOG_START.set(RenderSystem.getShaderFogStart());
		}
		if (FOG_END != null) {
			FOG_END.set(RenderSystem.getShaderFogEnd());
		}
		if (FOG_COLOR != null) {
			FOG_COLOR.set(RenderSystem.getShaderFogColor());
		}

		CapturedRenderingState.INSTANCE.setCurrentAlphaTest(alphaTest);

		// Phase 7: Set the active uniform buffer BEFORE any uniform updates
		// so that IrisRenderSystem.uniform*() writes to our buffer
		if (irisUniformBuffer != null) {
			IrisRenderSystem.setActiveUniformBuffer(irisUniformBuffer);
		}

		if (lastApplied != this) {
			lastApplied = this;
			ProgramManager.glUseProgram(this.getId());
		}

		if (intensitySwizzle) {
			IrisRenderSystem.texParameteriv(RenderSystem.getShaderTexture(0), TextureType.TEXTURE_2D.getGlType(), GL_TEXTURE_SWIZZLE_RGBA,
				new int[]{GL_RED, GL_RED, GL_RED, GL_RED});
		}

		IrisRenderSystem.bindTextureToUnit(TextureType.TEXTURE_2D.getGlType(), IrisSamplers.ALBEDO_TEXTURE_UNIT, RenderSystem.getShaderTexture(0));
		IrisRenderSystem.bindTextureToUnit(TextureType.TEXTURE_2D.getGlType(), IrisSamplers.OVERLAY_TEXTURE_UNIT, RenderSystem.getShaderTexture(1));
		IrisRenderSystem.bindTextureToUnit(TextureType.TEXTURE_2D.getGlType(), IrisSamplers.LIGHTMAP_TEXTURE_UNIT, RenderSystem.getShaderTexture(2));

		ImmediateState.usingTessellation = usesTessellation;

		if (PROJECTION_MATRIX != null) {
			// Always compute the inverse (needed for UBO even if MC Uniform is null)
			tempMatrix4f.set(PROJECTION_MATRIX.getFloatBuffer()).invert().get(tempFloats);
			if (projectionInverse != null) {
				projectionInverse.set(tempFloats);
			}
			if (irisUniformBuffer != null) {
				irisUniformBuffer.writeMat4f(irisUniformBuffer.getFieldOffset("iris_ProjMatInverse"), tempFloats);
			}
		} else {
			if (projectionInverse != null) {
				projectionInverse.set(identity);
			}
		}

		// Derived matrices from MODEL_VIEW_MATRIX (set from RenderSystem MV in apply())
		{
			org.joml.Matrix4f mv;
			if (MODEL_VIEW_MATRIX != null) {
				tempMatrix4f.set(MODEL_VIEW_MATRIX.getFloatBuffer());
				mv = tempMatrix4f;
			} else {
				mv = IDENTITY_MATRIX;
			}

			// ModelViewMatInverse
			org.joml.Matrix4f mvInv = new org.joml.Matrix4f(mv).invert();
			mvInv.get(tempFloats);
			if (modelViewInverse != null) {
				modelViewInverse.set(tempFloats);
			}
			if (irisUniformBuffer != null) {
				irisUniformBuffer.writeMat4f(irisUniformBuffer.getFieldOffset("iris_ModelViewMatInverse"), tempFloats);
			}

			// Normal matrix = transpose(inverse(mat3(ModelViewMat)))
			org.joml.Matrix3f normalMat = new org.joml.Matrix3f(mv).invert().transpose();
			normalMat.get(tempFloats2);
			if (normalMatrix != null) {
				normalMatrix.set(tempFloats2);
			}
			if (irisUniformBuffer != null) {
				irisUniformBuffer.writeMat3f(irisUniformBuffer.getFieldOffset("iris_NormalMat"), tempFloats2);
			}
		}

		uploadIfNotNull(projectionInverse);
		uploadIfNotNull(modelViewInverse);
		uploadIfNotNull(normalMatrix);

		// Phase 7: Write vanilla MC uniform values directly to UBO buffer
		// (VulkanMod's UniformM cancels Uniform.upload(), so we write manually)
		if (irisUniformBuffer != null) {
			writeMcUniformsToBuffer();
		}

		List<Uniform> uniformList = super.uniforms;
		for (Uniform uniform : uniformList) {
			uploadIfNotNull(uniform);
		}

		samplers.update();

		// This calls Iris Uniform.update() which now writes to the active buffer
		// via IrisRenderSystem.uniform*() methods
		uniforms.update();

		customUniforms.push(this);

		images.update();

		// Phase 7: Update ManualUBO source pointer so VulkanMod copies our data at draw time
		if (irisManualUBO != null && irisUniformBuffer != null) {
			irisManualUBO.setSrc(irisUniformBuffer.getPointer(), irisUniformBuffer.getUsedSize());
		}

		// In uniformsOnlyMode (called from VBO.drawWithShader for sky rendering):
		// - DO apply blend mode and bind gbuffer framebuffer so sky data reaches
		//   the gbuffer for composite passes to read
		// - SKIP pipeline binding because VBO.drawWithShader does its own pipeline
		//   binding via renderer.bindGraphicsPipeline()

		if (!uniformsOnlyMode) {
			if (this.blendModeOverride != null) {
				this.blendModeOverride.apply();
			}

			if (hasOverrides) {
				bufferBlendOverrides.forEach(BufferBlendOverride::apply);
			}
		}

		// Bind the gbuffer framebuffer, but skip if the same VulkanMod Framebuffer
		// is already bound (avoids unnecessary render pass restart).
		// With the global Framebuffer cache in GlFramebuffer, terrain and entity
		// framebuffers with identical attachments share the same VulkanMod Framebuffer,
		// so this check usually succeeds after terrain has already bound the gbuffer.
		{
			GlFramebuffer targetFb = parent.isBeforeTranslucent ? writingToBeforeTranslucent : writingToAfterTranslucent;
			net.vulkanmod.vulkan.framebuffer.Framebuffer vkFb = targetFb.getVulkanFramebuffer();
			net.vulkanmod.vulkan.Renderer vkRenderer = net.vulkanmod.vulkan.Renderer.getInstance();
			if (vkFb == null || vkRenderer.getBoundFramebuffer() != vkFb) {
				// Different framebuffer or not yet created — must bind (may restart render pass)
				targetFb.bind();
			}
			// If same framebuffer is already bound, render pass stays active — zero cost
		}

		// Skip pipeline binding in uniformsOnlyMode — VBO.drawWithShader handles it
		if (uniformsOnlyMode) {
			return;
		}

		// Bind the Vulkan pipeline for entity/shader rendering.
		// Without this, entities render with the default VulkanMod pipeline instead
		// of the Iris shader pack pipeline, so they don't write to gbuffer correctly.
		GraphicsPipeline vulkanPipeline = ((ShaderMixed) (Object) this).getPipeline();
		if (vulkanPipeline != null) {
			net.vulkanmod.vulkan.Renderer renderer = net.vulkanmod.vulkan.Renderer.getInstance();
			renderer.bindGraphicsPipeline(vulkanPipeline);
			renderer.uploadAndBindUBOs(vulkanPipeline);
		}
	}

	/**
	 * Updates only the UBO uniform data without binding framebuffers or pipelines.
	 * Called from VBO.drawWithShader so that iris_ModelViewMat etc. reflect the
	 * correct MV/P matrices during sky rendering, without disrupting the active
	 * Vulkan render pass. Implements ShaderMixed.updateUniformsOnly().
	 */
	public void updateUniformsOnly() {
		uniformsOnlyMode = true;
		try {
			apply();
		} finally {
			uniformsOnlyMode = false;
		}
	}

	/**
	 * Writes standard Minecraft uniform values directly to the IrisUniformBuffer.
	 * These uniforms (ModelViewMat, ProjMat, ColorModulator, etc.) may be declared
	 * in shader pack GLSL and thus present in the IrisUniforms UBO block.
	 * VulkanMod's UniformM cancels Uniform.upload(), so we write them manually.
	 */

	// Identity matrix as float array for UBO writes
	private static final float[] IDENTITY_MAT4 = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	private void writeMcUniformsToBuffer() {
		// The GLSL transformer renames gl_ModelViewMatrix → iris_ModelViewMat,
		// gl_ProjectionMatrix → iris_ProjMat, etc. So the UBO field may have
		// the iris_ prefix. Try both vanilla and iris-prefixed names.

		// ModelViewMat / iris_ModelViewMat — from RenderSystem.getModelViewMatrix()
		// VulkanMod's entity culling (LevelRendererM) renders entities with an identity
		// PoseStack, so entity vertices are in camera-relative world space (NOT camera-rotated).
		// The camera rotation is in RenderSystem.getModelViewMatrix() and must be applied
		// by the shader to transform vertices to view space.
		if (MODEL_VIEW_MATRIX != null) {
			java.nio.FloatBuffer mvBuf = MODEL_VIEW_MATRIX.getFloatBuffer();
			writeToFirstMatch(mvBuf, "ModelViewMat", "iris_ModelViewMat", "iris_ModelViewMatrix");
		}
		// ProjMat / iris_ProjMat
		if (PROJECTION_MATRIX != null) {
			java.nio.FloatBuffer projBuf = PROJECTION_MATRIX.getFloatBuffer();
			writeToFirstMatch(projBuf, "ProjMat", "iris_ProjMat", "iris_ProjectionMatrix");
		}
		// TextureMat / iris_TextureMat — from RenderSystem.getTextureMatrix()
		// The shader uses texCoord = (iris_TextureMat * gl_MultiTexCoord0).xy
		// Without this, entity texture coordinates are garbage → wrong colors → bloom artifacts
		if (TEXTURE_MATRIX != null) {
			java.nio.FloatBuffer texMatBuf = TEXTURE_MATRIX.getFloatBuffer();
			writeToFirstMatch(texMatBuf, "TextureMat", "iris_TextureMat");
		} else {
			// Fallback: write identity so the shader gets valid texture coordinates
			for (String name : new String[]{"TextureMat", "iris_TextureMat"}) {
				int off = irisUniformBuffer.getFieldOffset(name);
				if (off >= 0) irisUniformBuffer.writeMat4f(off, IDENTITY_MAT4);
			}
		}
		// ColorModulator / iris_ColorModulator
		if (COLOR_MODULATOR != null) {
			java.nio.FloatBuffer fb = COLOR_MODULATOR.getFloatBuffer();
			float r = fb.get(0), g = fb.get(1), b = fb.get(2), a = fb.get(3);
			for (String name : new String[]{"ColorModulator", "iris_ColorModulator"}) {
				int off = irisUniformBuffer.getFieldOffset(name);
				if (off >= 0) irisUniformBuffer.writeVec4f(off, r, g, b, a);
			}
		}
		// FogStart / iris_FogStart
		{
			float val = RenderSystem.getShaderFogStart();
			for (String name : new String[]{"FogStart", "iris_FogStart"}) {
				int off = irisUniformBuffer.getFieldOffset(name);
				if (off >= 0) irisUniformBuffer.writeFloat(off, val);
			}
		}
		// FogEnd / iris_FogEnd
		{
			float val = RenderSystem.getShaderFogEnd();
			for (String name : new String[]{"FogEnd", "iris_FogEnd"}) {
				int off = irisUniformBuffer.getFieldOffset(name);
				if (off >= 0) irisUniformBuffer.writeFloat(off, val);
			}
		}
		// FogColor / iris_FogColor — use Iris's captured fog color for correct timing
		{
			org.joml.Vector3d capturedFog = CapturedRenderingState.INSTANCE.getFogColor();
			float fogR = (float) capturedFog.x;
			float fogG = (float) capturedFog.y;
			float fogB = (float) capturedFog.z;
			for (String name : new String[]{"FogColor", "iris_FogColor"}) {
				int off = irisUniformBuffer.getFieldOffset(name);
				if (off >= 0) irisUniformBuffer.writeVec4f(off, fogR, fogG, fogB, 1.0f);
			}
		}
		// GameTime
		{
			float val = RenderSystem.getShaderGameTime();
			int off = irisUniformBuffer.getFieldOffset("GameTime");
			if (off >= 0) irisUniformBuffer.writeFloat(off, val);
		}

		// Write gbufferModelView, gbufferProjection, etc. from CapturedRenderingState
		// These are the OptiFine-compatible uniforms used by shader packs for
		// camera-space calculations in both gbuffer and composite shaders.
		writeGbufferUniforms();

		// Ensure iris_ChunkOffset = (0,0,0) for entity/hand/particle rendering.
		// In OpenGL Iris, MC sets chunkOffset via glUniform3f per draw.
		// In Vulkan, Uniform.upload() is cancelled by VulkanMod, so the UBO field
		// could retain stale values. Entities use absolute coordinates (camera-relative),
		// so ChunkOffset must be zero. Terrain uses push constants for this instead.
		{
			int off = irisUniformBuffer.getFieldOffset("iris_ChunkOffset");
			if (off >= 0) irisUniformBuffer.writeVec3f(off, 0.0f, 0.0f, 0.0f);
		}

		// iris_ScreenSize — used by iris_widen_lines() for block outline rendering.
		// In OpenGL Iris, MC sets this via glUniform2f when the viewport changes.
		// In Vulkan, Uniform.upload() is cancelled, so the UBO retains the default
		// value of (1, 1) from ShaderCreator. With (1,1), the line widening formula
		// computes NDC offsets of ~lineWidth instead of ~lineWidth/screenSize,
		// causing block outlines to span the entire screen as thin radiating lines.
		{
			var window = Minecraft.getInstance().getWindow();
			if (window != null) {
				float w = (float) window.getWidth();
				float h = (float) window.getHeight();
				for (String name : new String[]{"ScreenSize", "iris_ScreenSize"}) {
					int off = irisUniformBuffer.getFieldOffset(name);
					if (off >= 0) irisUniformBuffer.writeVec2f(off, w, h);
				}
			}
		}

	}

	/**
	 * Write a mat4 FloatBuffer to the first matching UBO field name.
	 */
	private void writeToFirstMatch(java.nio.FloatBuffer matrix, String... names) {
		for (String name : names) {
			int off = irisUniformBuffer.getFieldOffset(name);
			if (off >= 0) {
				irisUniformBuffer.writeMat4f(off, matrix);
			}
		}
	}

	// Previous-frame matrix tracking (shared across all ExtendedShader instances).
	// Frame boundary detection: CapturedRenderingState.setGbufferProjection()
	// creates a new Matrix4f each frame, so reference comparison detects boundaries.
	private static float[] esPrevMvArr = null;
	private static float[] esPrevProjArr = null;
	private static float[] esSavedMvArr = null;
	private static float[] esSavedProjArr = null;
	private static org.joml.Matrix4fc esLastSeenProj = null;

	/**
	 * Writes gbufferModelView, gbufferProjection, their inverses, previous-frame
	 * matrices, shadow matrices, celestial positions, and other standard
	 * OptiFine/Iris uniforms to the UBO.
	 * Matches Program.writeGbufferUniforms() for consistency, except:
	 * - Does NOT negate m11 (entity gbuffer uses viewport Y-flip, not projection negate)
	 */
	private void writeGbufferUniforms() {
		org.joml.Matrix4fc gbufferMV = CapturedRenderingState.INSTANCE.getGbufferModelView();
		org.joml.Matrix4fc gbufferProj = CapturedRenderingState.INSTANCE.getGbufferProjection();

		// Detect frame boundary for previous-frame matrix tracking
		if (gbufferProj != null && gbufferProj != esLastSeenProj) {
			esPrevMvArr = esSavedMvArr;
			esPrevProjArr = esSavedProjArr;
			esSavedMvArr = null;
			esSavedProjArr = null;
			esLastSeenProj = gbufferProj;
		}

		if (gbufferMV != null) {
			float[] arr = new float[16];
			gbufferMV.get(arr);
			writeUboMat("gbufferModelView", arr);
			if (esSavedMvArr == null) esSavedMvArr = arr.clone();
			// Inverse
			org.joml.Matrix4f inv = new org.joml.Matrix4f(gbufferMV).invert();
			inv.get(arr);
			writeUboMat("gbufferModelViewInverse", arr);
			// Previous frame
			if (esPrevMvArr != null) {
				writeUboMat("gbufferPreviousModelView", esPrevMvArr);
			} else {
				gbufferMV.get(arr);
				writeUboMat("gbufferPreviousModelView", arr);
			}
		}

		if (gbufferProj != null) {
			org.joml.Matrix4f proj = new org.joml.Matrix4f(gbufferProj);

			// Fix infinite m00/m11 from VulkanMod's infinite far plane
			if (proj.m23() != 0) { // perspective projection
				Minecraft mcClient = Minecraft.getInstance();
				float far = mcClient.gameRenderer != null ? mcClient.gameRenderer.getRenderDistance() : 256.0f;
				float near = 0.05f;

				if (!Float.isFinite(proj.m00()) || !Float.isFinite(proj.m11())) {
					double fovDegrees = 70.0;
					try {
						if (mcClient.gameRenderer != null) {
							fovDegrees = ((net.irisshaders.iris.mixin.GameRendererAccessor) mcClient.gameRenderer)
								.invokeGetFov(mcClient.gameRenderer.getMainCamera(),
									mcClient.getTimer().getGameTimeDeltaPartialTick(true), true);
						}
					} catch (Exception ignored) {}
					if (fovDegrees < 1.0 || !Double.isFinite(fovDegrees)) fovDegrees = 70.0;
					float fovRad = (float)(fovDegrees * Math.PI / 180.0);
					float tanHalfFov = (float) Math.tan(fovRad / 2.0);
					var window = mcClient.getWindow();
					float aspect = (float) window.getWidth() / (float) window.getHeight();
					proj.m00(1.0f / (aspect * tanHalfFov));
					proj.m11(1.0f / tanHalfFov);
				}

				// Fix infinite-far depth elements with finite far
				proj.m22(-far / (far - near));
				proj.m32(-far * near / (far - near));
			}

			// Convert Vulkan [0,1] → OpenGL [-1,1] depth range
			proj.m22(2.0f * proj.m22() - proj.m23());
			proj.m32(2.0f * proj.m32() - proj.m33());
			// Do NOT negate m11 — gbuffer/entity passes use viewport Y-flip.
			// Composite passes handle Y mismatch via iris_flipProjY() in the shader.

			float[] arr = new float[16];
			proj.get(arr);
			writeUboMat("gbufferProjection", arr);
			if (esSavedProjArr == null) esSavedProjArr = arr.clone();
			// Inverse
			org.joml.Matrix4f inv = new org.joml.Matrix4f(proj).invert();
			inv.get(arr);
			writeUboMat("gbufferProjectionInverse", arr);
			// Previous frame
			if (esPrevProjArr != null) {
				writeUboMat("gbufferPreviousProjection", esPrevProjArr);
			} else {
				proj.get(arr);
				writeUboMat("gbufferPreviousProjection", arr);
			}
		}

		// cameraPosition
		Minecraft mc = Minecraft.getInstance();
		if (mc.gameRenderer != null && mc.gameRenderer.getMainCamera() != null) {
			var camPos = mc.gameRenderer.getMainCamera().getPosition();
			writeUboVec3("cameraPosition", (float) camPos.x, (float) camPos.y, (float) camPos.z);
		}

		// viewWidth / viewHeight
		if (mc.getWindow() != null) {
			writeUboFloat("viewWidth", (float) mc.getWindow().getWidth());
			writeUboFloat("viewHeight", (float) mc.getWindow().getHeight());
		}

		// near / far
		writeUboFloat("near", 0.05f);
		if (mc.gameRenderer != null) {
			writeUboFloat("far", mc.gameRenderer.getRenderDistance());
		}

		// Shadow matrices from ShadowRenderer
		org.joml.Matrix4f shadowMV = net.irisshaders.iris.shadows.ShadowRenderer.MODELVIEW;
		org.joml.Matrix4f shadowProj = net.irisshaders.iris.shadows.ShadowRenderer.PROJECTION;
		if (shadowMV != null) {
			float[] arr = new float[16];
			shadowMV.get(arr);
			writeUboMat("shadowModelView", arr);
			new org.joml.Matrix4f(shadowMV).invert().get(arr);
			writeUboMat("shadowModelViewInverse", arr);
		}
		if (shadowProj != null) {
			// ShadowMatrices.createOrthoMatrix() uses raw column values (NOT .ortho()),
			// so VulkanMod's Matrix4fM mixin does NOT affect it — already OpenGL-style.
			// Do NOT apply depth conversion or m11 negate.
			float[] arr = new float[16];
			shadowProj.get(arr);
			writeUboMat("shadowProjection", arr);
			new org.joml.Matrix4f(shadowProj).invert().get(arr);
			writeUboMat("shadowProjectionInverse", arr);
		}

		// Celestial light positions — critical for entity fragment lighting.
		// Without sunPosition, shaders compute sunVec = normalize(vec3(0)) = NaN.
		if (gbufferMV != null && mc.level != null) {
			float tickDelta = CapturedRenderingState.INSTANCE.getTickDelta();
			float skyAngle = mc.level.getTimeOfDay(tickDelta);
			float sunAngle = skyAngle < 0.75f ? skyAngle + 0.25f : skyAngle - 0.75f;

			float sunPathRotation = 0.0f;
			try {
				var pm = net.irisshaders.iris.Iris.getPipelineManager();
				if (pm != null) {
					var wp = pm.getPipelineNullable();
					if (wp instanceof net.irisshaders.iris.pipeline.IrisRenderingPipeline irp) {
						sunPathRotation = irp.getSunPathRotation();
					}
				}
			} catch (Exception ignored) {}

			org.joml.Matrix4f celestial = new org.joml.Matrix4f(gbufferMV);
			celestial.rotateY((float) Math.toRadians(-90.0));
			celestial.rotateZ((float) Math.toRadians(sunPathRotation));
			celestial.rotateX((float) Math.toRadians(skyAngle * 360.0f));

			org.joml.Vector4f sunPos = new org.joml.Vector4f(0, 100, 0, 0);
			celestial.transform(sunPos);
			writeUboVec3("sunPosition", sunPos.x(), sunPos.y(), sunPos.z());

			org.joml.Vector4f moonPos = new org.joml.Vector4f(0, -100, 0, 0);
			celestial.transform(moonPos);
			writeUboVec3("moonPosition", moonPos.x(), moonPos.y(), moonPos.z());

			boolean isDay = sunAngle <= 0.5f;
			if (isDay) {
				writeUboVec3("shadowLightPosition", sunPos.x(), sunPos.y(), sunPos.z());
			} else {
				writeUboVec3("shadowLightPosition", moonPos.x(), moonPos.y(), moonPos.z());
			}

			writeUboFloat("sunAngle", sunAngle);
			float shadowAngle = isDay ? sunAngle : sunAngle - 0.5f;
			writeUboFloat("shadowAngle", shadowAngle);

			// upPosition: modelView * rotY(-90) * (0, 100, 0, 0)
			org.joml.Matrix4f preCelestial = new org.joml.Matrix4f(gbufferMV);
			preCelestial.rotateY((float) Math.toRadians(-90.0));
			org.joml.Vector4f upPos = new org.joml.Vector4f(0, 100, 0, 0);
			preCelestial.transform(upPos);
			writeUboVec3("upPosition", upPos.x(), upPos.y(), upPos.z());
		}
	}

	private void writeUboMat(String name, float[] arr) {
		int off = irisUniformBuffer.getFieldOffset(name);
		if (off >= 0) irisUniformBuffer.writeMat4f(off, arr);
	}

	private void writeUboVec3(String name, float x, float y, float z) {
		int off = irisUniformBuffer.getFieldOffset(name);
		if (off >= 0) irisUniformBuffer.writeVec3f(off, x, y, z);
	}

	private void writeUboFloat(String name, float val) {
		int off = irisUniformBuffer.getFieldOffset(name);
		if (off >= 0) irisUniformBuffer.writeFloat(off, val);
	}

	@Nullable
	@Override
	public Uniform getUniform(@NotNull String name) {
		// Prefix all uniforms with Iris to help avoid conflicts with existing names within the shader.
		return super.getUniform("iris_" + name);
	}

	private void uploadIfNotNull(Uniform uniform) {
		if (uniform != null) {
			uniform.upload();
		}
	}

	@Override
	public void attachToProgram() {
		super.attachToProgram();
		if (this.geometry != null) {
			this.geometry.attachToShader(this);
		}
		if (this.tessControl != null) {
			this.tessControl.attachToShader(this);
		}
		if (this.tessEval != null) {
			this.tessEval.attachToShader(this);
		}
	}

	@Override
	public void iris$createExtraShaders(ResourceProvider factory, String name) {
		factory.getResource(ResourceLocation.fromNamespaceAndPath("minecraft", name + "_geometry.gsh")).ifPresent(geometry -> {
			try {
				this.geometry = Program.compileShader(IrisProgramTypes.GEOMETRY, name, geometry.open(), geometry.sourcePackId(), new GlslPreprocessor() {
					@Nullable
					@Override
					public String applyImport(boolean bl, String string) {
						return null;
					}
				});
				if (this.geometry != null) GLDebug.nameObject(GL_SHADER, this.geometry.getId(), name + "_geometry.gsh");
			} catch (IOException e) {
				Iris.logger.error("Failed to create shader program", e);
			}
		});
		factory.getResource(ResourceLocation.fromNamespaceAndPath("minecraft", name + "_tessControl.tcs")).ifPresent(tessControl -> {
			try {
				this.tessControl = Program.compileShader(IrisProgramTypes.TESS_CONTROL, name, tessControl.open(), tessControl.sourcePackId(), new GlslPreprocessor() {
					@Nullable
					@Override
					public String applyImport(boolean bl, String string) {
						return null;
					}
				});
				if (this.tessControl != null) GLDebug.nameObject(GL_SHADER, this.tessControl.getId(), name + "_tessControl.tcs");
			} catch (IOException e) {
				Iris.logger.error("Failed to create shader program", e);
			}
		});
		factory.getResource(ResourceLocation.fromNamespaceAndPath("minecraft", name + "_tessEval.tes")).ifPresent(tessEval -> {
			try {
				this.tessEval = Program.compileShader(IrisProgramTypes.TESS_EVAL, name, tessEval.open(), tessEval.sourcePackId(), new GlslPreprocessor() {
					@Nullable
					@Override
					public String applyImport(boolean bl, String string) {
						return null;
					}
				});
				if (this.tessEval != null) GLDebug.nameObject(GL_SHADER, this.tessEval.getId(), name + "_tessEval.tes");
			} catch (IOException e) {
				Iris.logger.error("Failed to create shader program", e);
			}
		});
	}

	public Program getGeometry() {
		return this.geometry;
	}

	public Program getTessControl() {
		return this.tessControl;
	}

	public Program getTessEval() {
		return this.tessEval;
	}

	public boolean hasActiveImages() {
		return images.getActiveImages() > 0;
	}

	// ==================== Vulkan Pipeline Creation ====================

	private static final Pattern SAMPLER_PATTERN = Pattern.compile(
		"^\\s*(?:layout\\s*\\([^)]*\\)\\s*)?uniform\\s+((?:sampler|isampler|usampler)\\w+)\\s+(\\w+)");

	// Pattern for vertex input declarations: [qualifiers] in TYPE NAME;
	// Group 1: optional qualifiers before "in" (flat, smooth, noperspective)
	// Group 2: type (vec3, ivec2, etc.)
	// Group 3: name (Position, iris_Entity, etc.)
	private static final Pattern VERTEX_INPUT_PATTERN = Pattern.compile(
		"^\\s*((?:(?:flat|smooth|noperspective)\\s+)*)in\\s+(\\w+)\\s+(\\w+)\\s*;");

	/**
	 * Creates a VulkanMod GraphicsPipeline for this ExtendedShader.
	 * Uses the shared uniform list (computed earlier in constructor) to ensure
	 * vertex and fragment shaders share the same UBO layout.
	 * Uses ManualUBO for uniform data upload from our IrisUniformBuffer.
	 */
	private void createVulkanPipeline(ResourceProvider resourceFactory, String name, VertexFormat vertexFormat, java.util.Map<String, Integer> samplerUnitMap) {
		try {
			// 1. Read shader sources from resource provider
			String vshSrc = readShaderSource(resourceFactory, ".vsh");
			String fshSrc = readShaderSource(resourceFactory, ".fsh");
			if (vshSrc == null || fshSrc == null) {
				Iris.logger.warn("Could not read shader sources for Vulkan pipeline: {}", name);
				return;
			}

			// 2. Preprocess for Vulkan using SHARED uniform list (same UBO layout for both stages)
			String vshVulkan;
			String fshVulkan;
			if (this.sharedUniforms != null) {
				vshVulkan = IrisSPIRVCompiler.prepareForVulkan(vshSrc, this.sharedUniforms);
				fshVulkan = IrisSPIRVCompiler.prepareForVulkan(fshSrc, this.sharedUniforms);
			} else {
				vshVulkan = IrisSPIRVCompiler.prepareForVulkan(vshSrc);
				fshVulkan = IrisSPIRVCompiler.prepareForVulkan(fshSrc);
			}

			// 2b. Patch vertex shader: convert gl_Position depth from GL [-1,1] to Vulkan [0,1].
			// iris_ProjMat is kept in GL convention for shader pack compatibility, but
			// Vulkan's rasterizer clips to [0,1] depth. Without this conversion,
			// near-plane clipping is wrong and causes vertex explosions for entities.
			vshVulkan = patchVertexShaderDepthRange(vshVulkan);

			// 3. Collect all unique sampler names from both shaders (in declaration order)
			List<String> allSamplers = new ArrayList<>();
			collectSamplerNames(vshVulkan, allSamplers);
			collectSamplerNames(fshVulkan, allSamplers);
			List<String> uniqueSamplers = new ArrayList<>(new LinkedHashSet<>(allSamplers));

			// 4. Create sampler binding map: binding 1, 2, 3... (UBO is at binding 0)
			Map<String, Integer> samplerBindings = new LinkedHashMap<>();
			for (int i = 0; i < uniqueSamplers.size(); i++) {
				samplerBindings.put(uniqueSamplers.get(i), i + 1);
			}

			// 5. Add explicit binding annotations to both shaders
			String vshFinal = addExplicitBindings(vshVulkan, samplerBindings);
			String fshFinal = addExplicitBindings(fshVulkan, samplerBindings);

			// 5b. Add explicit vertex input locations to ensure pipeline-shader location match
			// Without this, shaderc auto_map_locations assigns by declaration order which
			// may differ from the VertexFormat element order used by the Vulkan pipeline
			vshFinal = addExplicitInputLocations(vshFinal, vertexFormat);

			// Dump ALL ExtendedShader programs to iris-debug/ for inspection
			dumpEntityShader(name + ".vsh", vshFinal);
			dumpEntityShader(name + ".fsh", fshFinal);

			// 6. Compile to SPIR-V (source is already preprocessed, use compilePreprocessed)
			ByteBuffer vertSpirv = IrisSPIRVCompiler.compilePreprocessed(name + ".vsh", vshFinal, ShaderType.VERTEX);
			ByteBuffer fragSpirv = IrisSPIRVCompiler.compilePreprocessed(name + ".fsh", fshFinal, ShaderType.FRAGMENT);

			// 7. Wrap in VulkanMod SPIRV objects
			SPIRVUtils.SPIRV vertSPIRV = new SPIRVUtils.SPIRV(0, vertSpirv);
			SPIRVUtils.SPIRV fragSPIRV = new SPIRVUtils.SPIRV(0, fragSpirv);

			// 8. Create descriptor layout: 1 ManualUBO + N samplers
			List<UBO> ubos = new ArrayList<>();
			List<ImageDescriptor> imageDescriptors = new ArrayList<>();

			// Phase 7: Use ManualUBO instead of empty UBO - this copies our IrisUniformBuffer
			// data to the GPU each frame via ManualUBO.update()
			int vkShaderStageAllGraphics = 0x0000001F; // VK_SHADER_STAGE_ALL_GRAPHICS
			int uboSizeInBytes = (irisUniformBuffer != null) ? Math.max(irisUniformBuffer.getUsedSize(), 16) : 4096;
			// ManualUBO constructor takes size in 4-byte units (multiplies by 4 internally)
			int uboSizeInWords = (uboSizeInBytes + 3) / 4;
			this.irisManualUBO = new ManualUBO(0, vkShaderStageAllGraphics, uboSizeInWords);
			if (irisUniformBuffer != null) {
				this.irisManualUBO.setSrc(irisUniformBuffer.getPointer(), irisUniformBuffer.getUsedSize());
			}
			ubos.add(this.irisManualUBO);

			// Sampler ImageDescriptors at bindings 1, 2, 3...
			for (int i = 0; i < uniqueSamplers.size(); i++) {
				String samplerName = uniqueSamplers.get(i);
				int textureIdx = samplerUnitMap.containsKey(samplerName)
					? samplerUnitMap.get(samplerName) : mapSamplerToTextureIndex(samplerName);
				imageDescriptors.add(new ImageDescriptor(i + 1, "sampler2D", samplerName, textureIdx));
			}

			// 9. Build the VulkanMod GraphicsPipeline
			Pipeline.Builder builder = new Pipeline.Builder(vertexFormat, name);
			builder.setUniforms(ubos, imageDescriptors);
			builder.setSPIRVs(vertSPIRV, fragSPIRV);
			GraphicsPipeline pipeline = builder.createGraphicsPipeline();

			// 10. Set pipeline on this shader instance (via VulkanMod's ShaderMixed mixin)
			((ShaderMixed) (Object) this).setPipeline(pipeline);

			Iris.logger.debug("Created Vulkan pipeline for ExtendedShader: {} ({} samplers, {} UBO bytes)",
				name, uniqueSamplers.size(), uboSizeInBytes);
		} catch (Exception e) {
			Iris.logger.error("Failed to create Vulkan pipeline for ExtendedShader '{}': {}", name, e.getMessage());
			e.printStackTrace();

			// Attempt fallback: use VulkanMod's built-in pipeline for this shader name.
			// This gives entities SOME rendering (vanilla look) instead of being invisible.
			try {
				String builtInJsonPath = String.format("/assets/vulkanmod/shaders/minecraft/core/%s/%s.json", name, name);
				if (Pipeline.class.getResourceAsStream(builtInJsonPath) != null) {
					String builtInPath = String.format("minecraft/core/%s/%s", name, name);
					Pipeline.Builder fallbackBuilder = new Pipeline.Builder(vertexFormat, builtInPath);
					fallbackBuilder.parseBindingsJSON();
					fallbackBuilder.compileShaders();
					GraphicsPipeline fallbackPipeline = fallbackBuilder.createGraphicsPipeline();
					((ShaderMixed) (Object) this).setPipeline(fallbackPipeline);
					Iris.logger.debug("Fell back to VulkanMod built-in pipeline for ExtendedShader '{}'", name);
				} else {
					Iris.logger.warn("No VulkanMod built-in pipeline available for fallback: '{}'", name);
				}
			} catch (Exception fallbackEx) {
				Iris.logger.error("Fallback pipeline also failed for '{}': {}", name, fallbackEx.getMessage());
			}
		}
	}

	/**
	 * Dumps entity/particle shader source to iris-debug/ for inspection.
	 */
	private static void dumpEntityShader(String filename, String source) {
		if (source == null) return;
		try {
			java.io.File dir = new java.io.File("iris-debug");
			if (!dir.exists()) dir.mkdirs();
			java.io.File file = new java.io.File(dir, "entity_" + filename + ".glsl");
			java.nio.file.Files.writeString(file.toPath(), source);
		} catch (Exception e) {
			Iris.logger.warn("[ExtendedShader] Failed to dump shader {}: {}", filename, e.getMessage());
		}
	}

	/**
	 * Patches a vertex shader to convert gl_Position.z from GL [-1,1] depth range
	 * to Vulkan [0,1] depth range after the main function sets gl_Position.
	 *
	 * iris_ProjMat in the UBO is GL convention (shader packs expect this), but
	 * Vulkan's rasterizer clips clip-space Z to [0,1]. Without this conversion,
	 * vertices near the camera get clipped (z_ndc < 0) and triangles spanning
	 * the near plane produce stretched/exploded geometry.
	 *
	 * The conversion: z_clip_vk = z_clip_gl * 0.5 + w_clip * 0.5
	 * This maps z_ndc from [-1,1] to [0,1] in clip space.
	 */
	private static String patchVertexShaderDepthRange(String source) {
		if (DIAG_VK_DEPTH_BYPASS) {
			return source;
		}

		// Rename void main() to void _iris_vk_main()
		String patched = source.replaceFirst(
			"void\\s+main\\s*\\(\\s*(void)?\\s*\\)",
			"void _iris_vk_main()");

		if (patched.equals(source)) {
			Iris.logger.warn("[IrisVulkan] Could not patch vertex shader depth range: void main() not found");
			return source;
		}

		// Append a new main() that calls the original and converts depth
		patched += "\n// [Iris Vulkan] Convert gl_Position from GL [-1,1] to Vulkan [0,1] depth\n";
		patched += "void main() {\n";
		patched += "    _iris_vk_main();\n";
		patched += "    gl_Position.z = gl_Position.z * 0.5 + gl_Position.w * 0.5;\n";
		patched += "}\n";

		return patched;
	}

	/**
	 * Reads shader source from the resource provider by file extension.
	 * IrisProgramResourceFactory serves source based on extension only.
	 */
	private static String readShaderSource(ResourceProvider provider, String extension) {
		try {
			Optional<Resource> resource = provider.getResource(
				ResourceLocation.fromNamespaceAndPath("minecraft", "shaders/core/iris_dummy" + extension));
			if (resource.isPresent()) {
				try (InputStream is = resource.get().open()) {
					return new String(is.readAllBytes(), StandardCharsets.UTF_8);
				}
			}
		} catch (Exception e) {
			// Source not available
		}
		return null;
	}

	/**
	 * Finds all sampler declarations in preprocessed GLSL and collects their names.
	 */
	private static void collectSamplerNames(String source, List<String> names) {
		for (String line : source.split("\n")) {
			Matcher m = SAMPLER_PATTERN.matcher(line);
			if (m.find()) {
				String samplerName = m.group(2);
				if (!names.contains(samplerName)) {
					names.add(samplerName);
				}
			}
		}
	}

	/**
	 * Collects sampler names from raw GLSL source into a Set.
	 * Used to register sampler names with IrisRenderSystem for getUniformLocation() checks.
	 */
	private static void collectSamplerNamesFromSource(String source, java.util.Set<String> names) {
		for (String line : source.split("\n")) {
			Matcher m = SAMPLER_PATTERN.matcher(line.trim());
			if (m.find()) {
				names.add(m.group(2));
			}
		}
	}

	/**
	 * Post-processes Vulkan GLSL to add explicit binding annotations.
	 * - Adds binding=0 to the IrisUniforms UBO block
	 * - Adds binding=N to each sampler declaration from the binding map
	 */
	private static String addExplicitBindings(String source, Map<String, Integer> samplerBindings) {
		String[] lines = source.split("\n", -1);
		List<String> output = new ArrayList<>();

		for (String line : lines) {
			String trimmed = line.trim();

			// Add binding=0 to IrisUniforms UBO.
			// Matrix transposition is handled at write time in IrisUniformBuffer.writeMat4f()
			// rather than via GLSL layout qualifier (column_major had no effect on shaderc output).
			if (trimmed.equals("layout(std140) uniform IrisUniforms {")) {
				output.add("layout(std140, binding = 0) uniform IrisUniforms {");
				continue;
			}

			// Check for sampler declaration and add binding
			Matcher m = SAMPLER_PATTERN.matcher(trimmed);
			if (m.find()) {
				String samplerType = m.group(1);
				String samplerName = m.group(2);
				Integer binding = samplerBindings.get(samplerName);
				if (binding != null) {
					// Extract any array suffix or trailing content after the name
					String fullLine = trimmed.substring(m.end());
					output.add("layout(binding = " + binding + ") uniform " + samplerType + " " + samplerName + fullLine);
					continue;
				}
			}

			output.add(line);
		}

		return String.join("\n", output);
	}

	/**
	 * Adds explicit layout(location=N) qualifiers to ALL vertex shader input declarations
	 * to ensure format-matched inputs get the correct pipeline locations, and extra
	 * Iris-injected inputs (iris_Entity, iris_entityInfo, etc.) get non-conflicting locations.
	 *
	 * Without explicit locations on ALL inputs, shaderc's auto_map_locations can assign
	 * location 0 to unmatched inputs, conflicting with our explicit location 0 assignment.
	 */
	private static String addExplicitInputLocations(String vshSource, VertexFormat vertexFormat) {
		// Build name -> location map from vertex format
		// Location = element index in getElements(), matching GraphicsPipeline
		Map<String, Integer> nameToLocation = new LinkedHashMap<>();

		List<String> formatNames = new ArrayList<>();
		vertexFormat.getElementAttributeNames().forEach(formatNames::add);

		for (int i = 0; i < formatNames.size(); i++) {
			nameToLocation.put(formatNames.get(i), i);
		}

		// Add common aliases used by Iris-transformed shaders
		addInputAlias(nameToLocation, "UV1", "iris_UV1");
		addInputAlias(nameToLocation, "Position", "iris_Position");
		addInputAlias(nameToLocation, "Color", "iris_Color");
		addInputAlias(nameToLocation, "UV0", "iris_UV0");
		addInputAlias(nameToLocation, "UV2", "iris_UV2");
		addInputAlias(nameToLocation, "Normal", "iris_Normal");
		// va* naming convention from legacy shader packs
		addInputAlias(nameToLocation, "Position", "vaPosition");
		addInputAlias(nameToLocation, "Color", "vaColor");
		addInputAlias(nameToLocation, "UV0", "vaUV0");
		addInputAlias(nameToLocation, "UV1", "vaUV1");
		addInputAlias(nameToLocation, "UV2", "vaUV2");
		addInputAlias(nameToLocation, "Normal", "vaNormal");

		// First pass: collect all in declarations and assign locations
		// Format-matched inputs: location = element index
		// Extra inputs (Iris-injected): location = formatElements.size() + sequential
		String[] lines = vshSource.split("\n", -1);
		List<String> output = new ArrayList<>();
		int nextExtraLocation = formatNames.size(); // start extra locations after format elements
		int assignedCount = 0;
		StringBuilder assignments = new StringBuilder();

		for (String line : lines) {
			String trimmed = line.trim();

			// Skip lines that already have layout qualifiers on in declarations
			if (trimmed.contains("layout") && trimmed.contains(" in ")) {
				output.add(line);
				continue;
			}

			Matcher m = VERTEX_INPUT_PATTERN.matcher(trimmed);
			if (m.find()) {
				String qualifiers = m.group(1); // "flat " etc. or ""
				String type = m.group(2);
				String name = m.group(3);

				Integer location = nameToLocation.get(name);
				if (location != null) {
					// Format-matched: use element's pipeline location
					output.add("layout(location = " + location + ") " + qualifiers + "in " + type + " " + name + ";");
					assignedCount++;
					assignments.append(String.format(" %s→%d", name, location));
				} else {
					// Extra Iris-injected input: assign beyond format element range
					output.add("layout(location = " + nextExtraLocation + ") " + qualifiers + "in " + type + " " + name + ";");
					assignments.append(String.format(" %s→%d*", name, nextExtraLocation));
					nextExtraLocation++;
					assignedCount++;
				}
				continue;
			}

			output.add(line);
		}

		return String.join("\n", output);
	}

	private static void addInputAlias(Map<String, Integer> map, String originalName, String alias) {
		Integer location = map.get(originalName);
		if (location != null) {
			map.putIfAbsent(alias, location);
		}
	}

	/**
	 * Maps Iris/OptiFine sampler names to VulkanMod texture slot indices.
	 * VTextureSelector supports indices 0-11.
	 */
	private static int mapSamplerToTextureIndex(String name) {
		return switch (name) {
			case "gtexture", "gcolor", "colortex0", "tex", "texture" -> 0;
			case "overlay", "iris_overlay" -> 1;
			case "lightmap" -> 2;
			case "normals", "gnormal", "colortex1" -> 3;
			case "specular", "gspecular", "colortex2", "gaux1" -> 4;
			case "shadow", "watershadow", "shadowtex0" -> 5;
			case "shadowtex1" -> 6;
			case "shadowcolor0", "shadowcolor", "colortex3", "gaux2" -> 7;
			case "shadowcolor1", "colortex4", "gaux3" -> 8;
			case "noisetex", "colortex7", "gaux4" -> 9;
			case "depthtex0", "colortex5" -> 10;
			case "depthtex1", "colortex6" -> 11;
			case "depthtex2", "colortex8" -> 12;
			default -> {
				Iris.logger.warn("[ExtendedShader] Unknown sampler '{}' in mapSamplerToTextureIndex, defaulting to 0", name);
				yield 0;
			}
		};
	}
}
