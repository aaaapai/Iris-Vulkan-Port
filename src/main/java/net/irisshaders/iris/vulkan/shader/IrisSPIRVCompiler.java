package net.irisshaders.iris.vulkan.shader;

import net.irisshaders.iris.gl.shader.ShaderType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.BufferUtils;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.lwjgl.util.shaderc.Shaderc.*;

import org.lwjgl.vulkan.VK12;

/**
 * Iris SPIR-V Compiler - Vulkan Port.
 *
 * Compiles transformed GLSL source code to SPIR-V bytecode using shaderc
 * (via LWJGL's shaderc bindings, same library VulkanMod uses).
 *
 * The compilation flow is:
 * 1. Iris transforms shader pack GLSL (TransformPatcher: version upgrade, layout qualifiers, legacy replacement)
 * 2. This compiler preprocesses the GLSL for Vulkan (wraps loose uniforms in UBOs)
 * 3. shaderc compiles the preprocessed GLSL to SPIR-V with auto-bind/auto-map enabled
 * 4. The SPIR-V bytecode is used to create VkShaderModules
 *
 * Key Vulkan requirements handled:
 * - Non-opaque uniforms (float, vec, mat, int, etc.) must be in uniform blocks
 * - #version must be 460 for SPIR-V target
 * - Bindings and locations are auto-assigned by shaderc when not explicit
 */
public class IrisSPIRVCompiler {
	private static final Logger LOGGER = LogManager.getLogger(IrisSPIRVCompiler.class);

	// Cache compiled SPIR-V by source hash to avoid recompilation
	private static final Map<Integer, ByteBuffer> spirvCache = new ConcurrentHashMap<>();

	// Shaderc compiler and options (initialized once, thread-safe)
	private static long compiler = 0;
	private static long options = 0;

	// Matches loose uniform declarations: [layout(...)] uniform [precision] type name[array] [= value];
	// Group 1: type, Group 2: name, Group 3: optional array part
	// Handles initializers like: uniform float x = 1.0; uniform bool flag = false;
	private static final Pattern LOOSE_UNIFORM_PATTERN = Pattern.compile(
		"^\\s*(?:layout\\s*\\([^)]*\\)\\s*)?uniform\\s+(?:(?:lowp|mediump|highp)\\s+)?(\\w+)\\s+(\\w+)(\\s*\\[[^]]*\\])?(?:\\s*=[^;]*)?\\s*;");

	/**
	 * Initializes the shaderc compiler with Vulkan 1.2 target and auto-bind/auto-map options.
	 */
	private static synchronized void ensureInitialized() {
		if (compiler != 0) return;

		compiler = shaderc_compiler_initialize();
		if (compiler == 0) {
			throw new RuntimeException("Failed to initialize shaderc compiler");
		}

		options = shaderc_compile_options_initialize();

		// Target Vulkan 1.2 (matches VulkanMod's SPIRVUtils configuration)
		shaderc_compile_options_set_target_env(options,
			shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
		//shaderc_compile_options_set_target_env(options, shaderc_env_version_vulkan_1_1, VK12.VK_API_VERSION_1_1);
        //shaderc_compile_options_set_generate_debug_info(options);

		// Performance optimization (matches VulkanMod)
		shaderc_compile_options_set_optimization_level(options,
			shaderc_optimization_level_performance);

		// Auto-assign binding numbers to uniforms/samplers without explicit layout(binding=N)
		shaderc_compile_options_set_auto_bind_uniforms(options, true);

		// Auto-assign location numbers to in/out without explicit layout(location=N)
		shaderc_compile_options_set_auto_map_locations(options, true);

		// Handle combined image samplers (sampler2D) for Vulkan SPIR-V
		shaderc_compile_options_set_auto_combined_image_sampler(options, true);

		shaderc_compile_options_set_vulkan_rules_relaxed(options, true);
		//shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_1);

		LOGGER.info("Iris SPIR-V compiler initialized (Vulkan 1.1, auto-bind, auto-map)");
	}

	/**
	 * Compiles GLSL source code to SPIR-V bytecode.
	 *
	 * The source is expected to be already transformed by Iris's TransformPatcher
	 * (version upgraded, layout qualifiers added, legacy functions replaced, etc.)
	 * This method applies additional Vulkan-specific preprocessing (UBO wrapping)
	 * before compiling with shaderc.
	 *
	 * @param name       Shader name (for error reporting, e.g. "composite.fsh")
	 * @param source     Transformed GLSL source code
	 * @param shaderType The type of shader (vertex, fragment, geometry, compute, etc.)
	 * @return ByteBuffer containing the compiled SPIR-V bytecode
	 * @throws RuntimeException if compilation fails
	 */
	public static ByteBuffer compile(String name, String source, ShaderType shaderType) {
		// Check cache first
		int cacheKey = source.hashCode() ^ shaderType.hashCode();
		ByteBuffer cached = spirvCache.get(cacheKey);
		if (cached != null) {
			LOGGER.debug("SPIR-V cache hit for shader: {} ({})", name, shaderType.name());
			return cached;
		}

		ensureInitialized();

		// Convert sampler2DShadow to sampler2D with manual comparison.
		// Vulkan requires VkSampler.compareEnable=true for hardware shadow comparison,
		// but VulkanMod's sampler creation doesn't support this. Manual step() comparison
		// provides correct shadow lookups without needing comparison samplers.
		source = convertShadowComparison(name, source);

		// Preprocess for Vulkan: wrap loose uniforms in UBO, fix #version
		String vulkanSource = prepareForVulkan(source);

		LOGGER.debug("Compiling SPIR-V for shader: {} ({})", name, shaderType.name());

		String extension = getExtension(shaderType);
		String filename = name.endsWith(extension) ? name : name + extension;
		int shaderKind = getShaderKind(shaderType);

		// Use heap-allocated ByteBuffers instead of MemoryStack (CharSequence variant)
		// because shader pack sources can exceed MemoryStack's ~64KB limit.
		// Source text must NOT be null-terminated: shaderc uses buffer.remaining() as
		// the size, so a null terminator byte would be treated as source content.
		// Filename and entry point ARE null-terminated (C string parameters).
		ByteBuffer sourceBuf = MemoryUtil.memUTF8(vulkanSource, false);
		ByteBuffer filenameBuf = MemoryUtil.memUTF8(filename);
		ByteBuffer entryPointBuf = MemoryUtil.memUTF8("main");

		long result;
		try {
			result = shaderc_compile_into_spv(compiler, sourceBuf, shaderKind, filenameBuf, entryPointBuf, options);
		} finally {
			MemoryUtil.memFree(entryPointBuf);
			MemoryUtil.memFree(filenameBuf);
			MemoryUtil.memFree(sourceBuf);
		}

		if (result == 0) {
			throw new RuntimeException("shaderc returned null for " + filename);
		}

		try {
			if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
				String errorMsg = shaderc_result_get_error_message(result);
				LOGGER.error("SPIR-V compilation failed for shader: {} ({})", name, shaderType.name());
				LOGGER.error("shaderc error:\n{}", errorMsg);

				// Log preprocessed source with context around error lines
				String[] dbgLines = vulkanSource.split("\n");
				java.util.Set<Integer> errorLineNums = new java.util.TreeSet<>();
				java.util.regex.Matcher errLineMatcher = java.util.regex.Pattern.compile(":(\\d+):").matcher(errorMsg);
				while (errLineMatcher.find()) {
					errorLineNums.add(Integer.parseInt(errLineMatcher.group(1)));
				}
				StringBuilder preview = new StringBuilder("Preprocessed GLSL source (first 20 lines + error context):\n");
				int firstLines = Math.min(20, dbgLines.length);
				for (int i = 0; i < firstLines; i++) {
					preview.append(String.format("  %3d: %s\n", i + 1, dbgLines[i]));
				}
				for (int errLine : errorLineNums) {
					int ctx = 5;
					int startCtx = Math.max(0, errLine - ctx - 1);
					int endCtx = Math.min(dbgLines.length, errLine + ctx);
					preview.append(String.format("  --- context around error line %d ---\n", errLine));
					for (int i = startCtx; i < endCtx; i++) {
						String marker = (i + 1 == errLine) ? ">>>" : "   ";
						preview.append(String.format("%s%3d: %s\n", marker, i + 1, dbgLines[i]));
					}
				}
				LOGGER.error(preview.toString());

				throw new RuntimeException("SPIR-V compilation failed for " + filename + ":\n" + errorMsg);
			}

			// Copy bytecode to our own buffer (shaderc result memory is freed in finally)
			ByteBuffer resultBytes = shaderc_result_get_bytes(result);
			ByteBuffer spirv = BufferUtils.createByteBuffer(resultBytes.remaining());
			spirv.put(resultBytes);
			spirv.flip();

			spirvCache.put(cacheKey, spirv);
			LOGGER.debug("Successfully compiled SPIR-V for shader: {} ({} bytes)", name, spirv.remaining());
			return spirv;
		} finally {
			shaderc_result_release(result);
		}
	}

	/**
	 * Preprocesses GLSL source for Vulkan SPIR-V compilation.
	 *
	 * Handles two critical Vulkan requirements:
	 * 1. Ensures #version 460 as the first directive (removes all existing #version lines)
	 * 2. Wraps loose non-opaque uniforms into a layout(std140) uniform block
	 *    (Vulkan disallows non-opaque uniforms outside blocks)
	 *
	 * Opaque types (samplers, images) are left as-is since Vulkan allows them
	 * outside blocks. Their bindings are handled by shaderc's auto-bind option.
	 *
	 * Note: In GLSL, the 'uniform' keyword can only appear at global scope
	 * (never inside functions, structs, or other blocks), so we can safely match
	 * uniform declarations without tracking brace depth. This avoids issues with
	 * braces inside block comments throwing off the counter.
	 *
	 * @param source The GLSL source (already transformed by TransformPatcher)
	 * @return Vulkan-compatible GLSL source
	 */
	public static String prepareForVulkan(String source) {
		String[] lines = source.split("\n", -1);
		List<String> outputLines = new ArrayList<>();
		List<String> uboMembers = new ArrayList<>();

		boolean inBlockComment = false;

		for (String line : lines) {
			String trimmed = line.trim();

			// Track multi-line block comment state
			// We need this to avoid matching uniforms inside /* ... */ comments
			if (inBlockComment) {
				if (trimmed.contains("*/")) {
					inBlockComment = false;
					// Check if there's code after the comment end
					String afterComment = trimmed.substring(trimmed.indexOf("*/") + 2).trim();
					if (afterComment.isEmpty()) {
						outputLines.add(line);
						continue;
					}
					// Process the part after the comment
					trimmed = afterComment;
				} else {
					outputLines.add(line);
					continue;
				}
			}

			// Check for block comment start on this line
			if (trimmed.contains("/*")) {
				// If the block comment also ends on this line, strip it for matching
				if (trimmed.contains("*/")) {
					// Single-line block comment - strip it for pattern matching
					String stripped = trimmed.replaceAll("/\\*.*?\\*/", " ").trim();
					if (stripped.isEmpty()) {
						outputLines.add(line);
						continue;
					}
					trimmed = stripped;
				} else {
					// Block comment starts but doesn't end - check if there's code before it
					String beforeComment = trimmed.substring(0, trimmed.indexOf("/*")).trim();
					inBlockComment = true;
					if (beforeComment.isEmpty()) {
						outputLines.add(line);
						continue;
					}
					trimmed = beforeComment;
				}
			}

			// Remove ALL #version directives (we insert #version 460 at the top)
			if (trimmed.startsWith("#version")) {
				continue;
			}

			// Try to match a loose uniform declaration (ends with ;, not a block)
			Matcher m = LOOSE_UNIFORM_PATTERN.matcher(trimmed);
			if (m.find()) {
				String type = m.group(1);
				String uniformName = m.group(2);
				String arrayPart = m.group(3);

				if (isSamplerOrImageType(type)) {
					// Opaque types (samplers, images) CAN be outside blocks in Vulkan GLSL
					// shaderc auto-bind assigns their binding numbers
					outputLines.add(line);
				} else {
					// Non-opaque types (float, vec, mat, int, etc.) MUST be in a block
					// Collect for wrapping in IrisUniforms UBO
					uboMembers.add("    " + type + " " + uniformName
						+ (arrayPart != null ? arrayPart : "") + ";");
				}
				continue;
			}

			// Vulkan GLSL builtins: gl_VertexID -> gl_VertexIndex, gl_InstanceID -> gl_InstanceIndex
			if (line.contains("gl_VertexID")) {
				line = line.replace("gl_VertexID", "gl_VertexIndex");
			}
			if (line.contains("gl_InstanceID")) {
				line = line.replace("gl_InstanceID", "gl_InstanceIndex");
			}

			// Default: pass through unchanged
			outputLines.add(line);
		}

		// Always insert #version 460 as the very first line
		outputLines.add(0, "#version 460");

		// Insert UBO block for collected non-opaque uniforms
		if (!uboMembers.isEmpty()) {
			// Find insertion point: after #version, #extension directives, and leading blank lines
			int insertIdx = 1; // after #version 460
			for (int i = 1; i < outputLines.size(); i++) {
				String t = outputLines.get(i).trim();
				if (t.startsWith("#extension") || t.isEmpty()) {
					insertIdx = i + 1;
				} else {
					break;
				}
			}

			List<String> uboBlock = new ArrayList<>();
			uboBlock.add("");
			uboBlock.add("layout(std140) uniform IrisUniforms {");
			uboBlock.addAll(uboMembers);
			uboBlock.add("};");
			outputLines.addAll(insertIdx, uboBlock);

			LOGGER.debug("Wrapped {} loose uniforms in IrisUniforms UBO", uboMembers.size());
		}

		return String.join("\n", outputLines);
	}

	/**
	 * Converts sampler2DShadow usage to sampler2D with manual depth comparison.
	 *
	 * In OpenGL, sampler2DShadow + texture() performs hardware depth comparison
	 * using the sampler's GL_TEXTURE_COMPARE_MODE setting. In Vulkan, this requires
	 * VkSampler.compareEnable=true, which VulkanMod doesn't support.
	 *
	 * This method replaces sampler2DShadow with sampler2D and wraps shadow texture()
	 * calls with manual step() comparison, producing equivalent results.
	 */
	private static String convertShadowComparison(String name, String source) {
		if (!source.contains("sampler2DShadow")) return source;

		// Step 1: Replace texture/textureLod calls on shadow samplers with helper calls.
		// Must happen BEFORE type replacement so we can identify which calls need conversion.
		// Pattern matches: texture( shadowtex0 , ...) or texture( shadowtex1 , ...)
		int beforeLen = source.length();
		source = source.replaceAll(
			"texture\\s*\\(\\s*(shadowtex[01])\\s*,",
			"iris_shadowComp($1,");
		source = source.replaceAll(
			"textureLod\\s*\\(\\s*(shadowtex[01])\\s*,",
			"iris_shadowCompLod($1,");
		// textureOffset requires compile-time constant offset — can't wrap in a function.
		// Leave textureOffset calls as-is; after sampler2DShadow → sampler2D conversion
		// they become regular textureOffset(sampler2D, vec2, ivec2) which is valid.
		// textureGrad similarly works fine with sampler2D.

		// Also handle shadow sampler used with shadow2D (already renamed to texture by CommonTransformer)
		// and shadow samplers named 'shadow' or 'watershadow'
		source = source.replaceAll(
			"texture\\s*\\(\\s*(shadow|watershadow)\\s*,",
			"iris_shadowComp($1,");

		int replacements = source.length() - beforeLen;

		// Step 2: Replace sampler2DShadow type with sampler2D
		source = source.replace("sampler2DShadow", "sampler2D");

		// Step 3: Inject ONLY the helper functions that are actually referenced.
		// This avoids compilation errors from unused helpers (e.g. bias overload
		// is invalid in vertex shaders since texture(sampler2D, vec2, float bias)
		// requires fragment-shader derivatives).
		StringBuilder helpers = new StringBuilder();
		helpers.append("\n// Iris Vulkan: Manual shadow comparison (VkSampler lacks compareEnable support)\n");

		if (source.contains("iris_shadowComp(")) {
			// Bilinear PCF: matches hardware sampler2DShadow behavior.
			// Hardware shadow comparison does step() on 4 neighboring texels,
			// then bilinearly interpolates the 4 results based on sub-texel position.
			// This gives smooth 0.0-1.0 values at shadow boundaries, unlike simple
			// averaging which only produces 5 discrete levels (0, 0.25, 0.5, 0.75, 1).
			helpers.append("float iris_shadowComp(sampler2D s, vec3 c) {\n");
			helpers.append("    c.z -= 0.001;\n");
			helpers.append("    vec2 texSize = vec2(textureSize(s, 0));\n");
			helpers.append("    vec2 texelSize = 1.0 / texSize;\n");
			helpers.append("    vec2 f = fract(c.xy * texSize + 0.5);\n");
			helpers.append("    float s00 = step(c.z, texture(s, c.xy + vec2(-0.5, -0.5) * texelSize).r);\n");
			helpers.append("    float s10 = step(c.z, texture(s, c.xy + vec2( 0.5, -0.5) * texelSize).r);\n");
			helpers.append("    float s01 = step(c.z, texture(s, c.xy + vec2(-0.5,  0.5) * texelSize).r);\n");
			helpers.append("    float s11 = step(c.z, texture(s, c.xy + vec2( 0.5,  0.5) * texelSize).r);\n");
			helpers.append("    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);\n");
			helpers.append("}\n");
		}
		if (source.contains("iris_shadowCompLod(")) {
			helpers.append("float iris_shadowCompLod(sampler2D s, vec3 c, float lod) {\n");
			helpers.append("    c.z -= 0.001;\n");
			helpers.append("    vec2 texSize = vec2(textureSize(s, 0));\n");
			helpers.append("    vec2 texelSize = 1.0 / texSize;\n");
			helpers.append("    vec2 f = fract(c.xy * texSize + 0.5);\n");
			helpers.append("    float s00 = step(c.z, textureLod(s, c.xy + vec2(-0.5, -0.5) * texelSize, lod).r);\n");
			helpers.append("    float s10 = step(c.z, textureLod(s, c.xy + vec2( 0.5, -0.5) * texelSize, lod).r);\n");
			helpers.append("    float s01 = step(c.z, textureLod(s, c.xy + vec2(-0.5,  0.5) * texelSize, lod).r);\n");
			helpers.append("    float s11 = step(c.z, textureLod(s, c.xy + vec2( 0.5,  0.5) * texelSize, lod).r);\n");
			helpers.append("    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);\n");
			helpers.append("}\n");
		}

		// Only inject if we have helpers to add
		if (helpers.length() > 80) {
			// Find end of #version line
			Matcher versionMatcher = Pattern.compile("#version\\s+\\d+[^\\n]*").matcher(source);
			if (versionMatcher.find()) {
				int insertPos = versionMatcher.end();
				// Skip past any extensions that must follow #version
				Pattern extPattern = Pattern.compile("\\n\\s*#extension[^\\n]*", Pattern.MULTILINE);
				Matcher extSearch = extPattern.matcher(source);
				int lastExtEnd = insertPos;
				while (extSearch.find(lastExtEnd)) {
					String between = source.substring(lastExtEnd, extSearch.start()).trim();
					if (between.isEmpty() || between.startsWith("//") || between.startsWith("\n")) {
						lastExtEnd = extSearch.end();
					} else {
						break;
					}
				}
				source = source.substring(0, lastExtEnd) + helpers.toString() + source.substring(lastExtEnd);
			}
		}

		if (replacements != 0) {
			LOGGER.info("[IrisSPIRVCompiler] Shadow comparison converted for {} ({} chars changed)", name, Math.abs(replacements));
		}

		return source;
	}

	/**
	 * Returns true if the GLSL type is an opaque type (sampler, image, atomic counter,
	 * or subpass input) that can legally exist outside a uniform block in Vulkan GLSL.
	 */
	private static boolean isSamplerOrImageType(String type) {
		return type.startsWith("sampler") || type.startsWith("isampler") || type.startsWith("usampler")
			|| type.startsWith("image") || type.startsWith("iimage") || type.startsWith("uimage")
			|| type.equals("atomic_uint") || type.startsWith("subpassInput");
	}

	/**
	 * Maps Iris ShaderType to shaderc shader kind constant.
	 */
	private static int getShaderKind(ShaderType shaderType) {
		return switch (shaderType) {
			case VERTEX -> shaderc_glsl_vertex_shader;
			case FRAGMENT -> shaderc_glsl_fragment_shader;
			case GEOMETRY -> shaderc_glsl_geometry_shader;
			case COMPUTE -> shaderc_glsl_compute_shader;
			case TESSELATION_CONTROL -> shaderc_glsl_tess_control_shader;
			case TESSELATION_EVAL -> shaderc_glsl_tess_evaluation_shader;
		};
	}

	/**
	 * Gets the file extension for a shader type.
	 */
	private static String getExtension(ShaderType shaderType) {
		return switch (shaderType) {
			case VERTEX -> ".vsh";
			case FRAGMENT -> ".fsh";
			case GEOMETRY -> ".gsh";
			case COMPUTE -> ".csh";
			case TESSELATION_CONTROL -> ".tcs";
			case TESSELATION_EVAL -> ".tes";
		};
	}

	/**
	 * Represents a non-opaque uniform field collected from shader source.
	 */
	public record UniformField(String type, String name, String arrayPart) {}

	/**
	 * Collects all non-opaque (non-sampler/image) loose uniform declarations from GLSL source.
	 * Used to build a shared IrisUniforms block across vertex and fragment shaders.
	 */
	public static List<UniformField> collectLooseUniforms(String source) {
		List<UniformField> fields = new ArrayList<>();
		boolean inBlockComment = false;

		for (String line : source.split("\n")) {
			String trimmed = line.trim();

			// Track block comments
			if (inBlockComment) {
				if (trimmed.contains("*/")) {
					inBlockComment = false;
					trimmed = trimmed.substring(trimmed.indexOf("*/") + 2).trim();
					if (trimmed.isEmpty()) continue;
				} else continue;
			}
			if (trimmed.contains("/*")) {
				if (trimmed.contains("*/")) {
					trimmed = trimmed.replaceAll("/\\*.*?\\*/", " ").trim();
					if (trimmed.isEmpty()) continue;
				} else {
					trimmed = trimmed.substring(0, trimmed.indexOf("/*")).trim();
					inBlockComment = true;
					if (trimmed.isEmpty()) continue;
				}
			}
			if (trimmed.startsWith("//")) continue;

			Matcher m = LOOSE_UNIFORM_PATTERN.matcher(trimmed);
			if (m.find()) {
				String type = m.group(1);
				String name = m.group(2);
				String arrayPart = m.group(3);
				if (!isSamplerOrImageType(type)) {
					fields.add(new UniformField(type, name, arrayPart));
				}
			}
		}
		return fields;
	}

	/**
	 * Merges uniform fields from multiple shader sources into a single deduplicated list.
	 * Preserves declaration order; when the same name appears in multiple sources,
	 * keeps the first occurrence.
	 */
	public static List<UniformField> mergeUniforms(List<UniformField>... sources) {
		LinkedHashMap<String, UniformField> merged = new LinkedHashMap<>();
		for (List<UniformField> source : sources) {
			for (UniformField field : source) {
				merged.putIfAbsent(field.name(), field);
			}
		}
		return new ArrayList<>(merged.values());
	}

	/**
	 * Preprocesses GLSL source for Vulkan using a pre-built shared uniform list.
	 * This ensures vertex and fragment shaders use the same UBO layout.
	 *
	 * @param source The GLSL source (already transformed by TransformPatcher)
	 * @param sharedUniforms The combined uniform list for the IrisUniforms UBO block
	 * @return Vulkan-compatible GLSL source
	 */
	public static String prepareForVulkan(String source, List<UniformField> sharedUniforms) {
		String[] lines = source.split("\n", -1);
		List<String> outputLines = new ArrayList<>();
		boolean inBlockComment = false;

		for (String line : lines) {
			String trimmed = line.trim();

			if (inBlockComment) {
				if (trimmed.contains("*/")) {
					inBlockComment = false;
					String afterComment = trimmed.substring(trimmed.indexOf("*/") + 2).trim();
					if (afterComment.isEmpty()) {
						outputLines.add(line);
						continue;
					}
					trimmed = afterComment;
				} else {
					outputLines.add(line);
					continue;
				}
			}

			if (trimmed.contains("/*")) {
				if (trimmed.contains("*/")) {
					String stripped = trimmed.replaceAll("/\\*.*?\\*/", " ").trim();
					if (stripped.isEmpty()) {
						outputLines.add(line);
						continue;
					}
					trimmed = stripped;
				} else {
					String beforeComment = trimmed.substring(0, trimmed.indexOf("/*")).trim();
					inBlockComment = true;
					if (beforeComment.isEmpty()) {
						outputLines.add(line);
						continue;
					}
					trimmed = beforeComment;
				}
			}

			if (trimmed.startsWith("#version")) {
				continue;
			}

			// Remove loose non-opaque uniform declarations (they go in the shared UBO)
			Matcher m = LOOSE_UNIFORM_PATTERN.matcher(trimmed);
			if (m.find()) {
				String type = m.group(1);
				if (isSamplerOrImageType(type)) {
					outputLines.add(line);
				}
				// Non-opaque uniforms are skipped - they'll be in the shared UBO block
				continue;
			}

			if (line.contains("gl_VertexID")) {
				line = line.replace("gl_VertexID", "gl_VertexIndex");
			}
			if (line.contains("gl_InstanceID")) {
				line = line.replace("gl_InstanceID", "gl_InstanceIndex");
			}

			outputLines.add(line);
		}

		outputLines.add(0, "#version 460");

		// Insert the shared UBO block
		if (sharedUniforms != null && !sharedUniforms.isEmpty()) {
			int insertIdx = 1;
			for (int i = 1; i < outputLines.size(); i++) {
				String t = outputLines.get(i).trim();
				if (t.startsWith("#extension") || t.isEmpty()) {
					insertIdx = i + 1;
				} else {
					break;
				}
			}

			List<String> uboBlock = new ArrayList<>();
			uboBlock.add("");
			uboBlock.add("layout(std140) uniform IrisUniforms {");
			for (UniformField f : sharedUniforms) {
				uboBlock.add("    " + f.type() + " " + f.name()
					+ (f.arrayPart() != null ? f.arrayPart() : "") + ";");
			}
			uboBlock.add("};");
			outputLines.addAll(insertIdx, uboBlock);

			LOGGER.debug("Inserted shared IrisUniforms UBO with {} fields", sharedUniforms.size());
		}

		return String.join("\n", outputLines);
	}

	/**
	 * Compiles GLSL source that has already been preprocessed for Vulkan.
	 * Skips the prepareForVulkan step (caller already handled it).
	 */
	public static ByteBuffer compilePreprocessed(String name, String preprocessedSource, ShaderType shaderType) {
		// Convert sampler2DShadow to manual comparison before compilation.
		// This is applied here (not just in compile()) because most shader paths
		// use compilePreprocessed() — composites, entities, terrain all call this.
		preprocessedSource = convertShadowComparison(name, preprocessedSource);

		int cacheKey = preprocessedSource.hashCode() ^ shaderType.hashCode();
		ByteBuffer cached = spirvCache.get(cacheKey);
		if (cached != null) {
			return cached;
		}

		ensureInitialized();

		LOGGER.debug("Compiling pre-processed SPIR-V for shader: {} ({})", name, shaderType.name());

		String extension = getExtension(shaderType);
		String filename = name.endsWith(extension) ? name : name + extension;
		int shaderKind = getShaderKind(shaderType);

		ByteBuffer sourceBuf = MemoryUtil.memUTF8(preprocessedSource, false);
		ByteBuffer filenameBuf = MemoryUtil.memUTF8(filename);
		ByteBuffer entryPointBuf = MemoryUtil.memUTF8("main");

		long result;
		try {
			result = shaderc_compile_into_spv(compiler, sourceBuf, shaderKind, filenameBuf, entryPointBuf, options);
		} finally {
			MemoryUtil.memFree(entryPointBuf);
			MemoryUtil.memFree(filenameBuf);
			MemoryUtil.memFree(sourceBuf);
		}

		if (result == 0) {
			throw new RuntimeException("shaderc returned null for " + filename);
		}

		try {
			if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
				String errorMsg = shaderc_result_get_error_message(result);
				LOGGER.error("SPIR-V compilation failed for shader: {} ({})", name, shaderType.name());
				LOGGER.error("shaderc error:\n{}", errorMsg);

				String[] dbgLines = preprocessedSource.split("\n");
				java.util.Set<Integer> errorLineNums = new java.util.TreeSet<>();
				java.util.regex.Matcher errLineMatcher = java.util.regex.Pattern.compile(":(\\d+):").matcher(errorMsg);
				while (errLineMatcher.find()) {
					errorLineNums.add(Integer.parseInt(errLineMatcher.group(1)));
				}
				StringBuilder preview = new StringBuilder("Preprocessed GLSL source (first 20 lines + error context):\n");
				int firstLines = Math.min(20, dbgLines.length);
				for (int i = 0; i < firstLines; i++) {
					preview.append(String.format("  %3d: %s\n", i + 1, dbgLines[i]));
				}
				for (int errLine : errorLineNums) {
					int ctx = 5;
					int startCtx = Math.max(0, errLine - ctx - 1);
					int endCtx = Math.min(dbgLines.length, errLine + ctx);
					preview.append(String.format("  --- context around error line %d ---\n", errLine));
					for (int i = startCtx; i < endCtx; i++) {
						String marker = (i + 1 == errLine) ? ">>>" : "   ";
						preview.append(String.format("%s%3d: %s\n", marker, i + 1, dbgLines[i]));
					}
				}
				LOGGER.error(preview.toString());

				throw new RuntimeException("SPIR-V compilation failed for " + filename + ":\n" + errorMsg);
			}

			ByteBuffer resultBytes = shaderc_result_get_bytes(result);
			ByteBuffer spirv = BufferUtils.createByteBuffer(resultBytes.remaining());
			spirv.put(resultBytes);
			spirv.flip();

			spirvCache.put(cacheKey, spirv);
			LOGGER.debug("Successfully compiled pre-processed SPIR-V for shader: {} ({} bytes)", name, spirv.remaining());
			return spirv;
		} finally {
			shaderc_result_release(result);
		}
	}

	/**
	 * Clears the SPIR-V compilation cache.
	 * Should be called when shader packs are reloaded.
	 */
	public static void clearCache() {
		LOGGER.debug("Clearing SPIR-V cache ({} entries)", spirvCache.size());
		spirvCache.clear();
	}

	/**
	 * Returns the number of cached SPIR-V compilations.
	 */
	public static int getCacheSize() {
		return spirvCache.size();
	}

	/**
	 * Standard Iris uniforms that shader packs expect to be available.
	 *
	 * In OpenGL Iris, these are injected at runtime via glGetUniformLocation() + callbacks
	 * from CelestialUniforms, CameraUniforms, ViewportUniforms, etc. Shader packs may
	 * reference them in code without declaring them as loose uniforms (relying on Iris's
	 * runtime injection), or the declarations may be in included files that get processed
	 * in ways that collectLooseUniforms() doesn't catch.
	 *
	 * In Vulkan, all non-opaque uniforms must be in the IrisUniforms UBO block, which is
	 * built from the collected loose uniforms. This list ensures critical uniforms are
	 * always present in the UBO regardless of whether the regex found them in the source.
	 */
	private static final List<UniformField> STANDARD_IRIS_UNIFORMS = List.of(
		// CelestialUniforms — CRITICAL for deferred lighting
		new UniformField("float", "sunAngle", null),
		new UniformField("float", "shadowAngle", null),
		new UniformField("vec3", "sunPosition", null),
		new UniformField("vec3", "moonPosition", null),
		new UniformField("vec3", "shadowLightPosition", null),
		new UniformField("vec3", "upPosition", null),
		// CameraUniforms
		new UniformField("vec3", "cameraPosition", null),
		new UniformField("vec3", "previousCameraPosition", null),
		new UniformField("float", "eyeAltitude", null),
		new UniformField("float", "near", null),
		new UniformField("float", "far", null),
		// ViewportUniforms
		new UniformField("float", "viewWidth", null),
		new UniformField("float", "viewHeight", null),
		new UniformField("float", "aspectRatio", null),
		// WorldTimeUniforms
		new UniformField("int", "worldTime", null),
		new UniformField("int", "worldDay", null),
		new UniformField("int", "moonPhase", null),
		// SystemTimeUniforms
		new UniformField("int", "frameCounter", null),
		new UniformField("float", "frameTime", null),
		new UniformField("float", "frameTimeCounter", null),
		// CommonUniforms — most commonly used by shader packs
		new UniformField("int", "isEyeInWater", null),
		new UniformField("float", "blindness", null),
		new UniformField("float", "nightVision", null),
		new UniformField("float", "screenBrightness", null),
		new UniformField("float", "rainStrength", null),
		new UniformField("float", "wetness", null),
		new UniformField("float", "playerMood", null),
		new UniformField("ivec2", "eyeBrightness", null),
		new UniformField("ivec2", "eyeBrightnessSmooth", null),
		// MatrixUniforms — critical for position reconstruction
		new UniformField("mat4", "gbufferModelView", null),
		new UniformField("mat4", "gbufferModelViewInverse", null),
		new UniformField("mat4", "gbufferProjection", null),
		new UniformField("mat4", "gbufferProjectionInverse", null),
		new UniformField("mat4", "gbufferPreviousModelView", null),
		new UniformField("mat4", "gbufferPreviousProjection", null),
		new UniformField("mat4", "shadowModelView", null),
		new UniformField("mat4", "shadowModelViewInverse", null),
		new UniformField("mat4", "shadowProjection", null),
		new UniformField("mat4", "shadowProjectionInverse", null)
	);

	/**
	 * Ensures standard Iris uniforms are present in the merged uniform list.
	 * Appends any missing standard uniforms that shader packs commonly need.
	 *
	 * This handles the gap between OpenGL Iris (runtime uniform injection via
	 * glGetUniformLocation) and Vulkan Iris (uniforms must be in UBO text).
	 *
	 * @param merged The merged uniform list from collectLooseUniforms + mergeUniforms
	 * @return List of uniform names that were injected (empty if none)
	 */
	public static List<String> ensureStandardIrisUniforms(List<UniformField> merged) {
		Set<String> existingNames = new HashSet<>();
		for (UniformField f : merged) {
			existingNames.add(f.name());
		}

		List<String> injected = new ArrayList<>();
		for (UniformField standard : STANDARD_IRIS_UNIFORMS) {
			if (!existingNames.contains(standard.name())) {
				merged.add(standard);
				injected.add(standard.name());
			}
		}
		return injected;
	}

	// ==================== SPIR-V UBO Layout Parser ====================

	/**
	 * Parses a compiled SPIR-V binary to extract UBO member offsets, names,
	 * and matrix layout decorations (RowMajor vs ColMajor).
	 *
	 * This is used to verify that our Java-computed std140 offsets match
	 * what shaderc actually generated in the SPIR-V bytecode.
	 *
	 * @param spirv The compiled SPIR-V bytecode
	 * @param shaderName Name for logging purposes
	 * @return A SPIRVLayout describing the UBO, or null if parsing fails
	 */
	public static SPIRVLayout parseSPIRVLayout(ByteBuffer spirv, String shaderName) {
		try {
			int savedPos = spirv.position();
			int savedLimit = spirv.limit();

			// SPIR-V header: 5 words (20 bytes)
			if (spirv.remaining() < 20) return null;

			int magic = spirv.getInt(savedPos);
			if (magic != 0x07230203) {
				LOGGER.warn("[SPIRV_PARSE] Invalid magic: 0x{} for {}", Integer.toHexString(magic), shaderName);
				return null;
			}

			// Scan instructions for decorations and names
			Map<Integer, Map<Integer, String>> memberNames = new LinkedHashMap<>(); // typeId -> (memberIdx -> name)
			Map<Integer, Map<Integer, Integer>> memberOffsets = new LinkedHashMap<>(); // typeId -> (memberIdx -> offset)
			Set<Integer> blockTypeIds = new LinkedHashSet<>(); // type IDs decorated with Block
			Map<Integer, Map<Integer, Boolean>> memberRowMajor = new LinkedHashMap<>(); // typeId -> (memberIdx -> isRowMajor)
			Map<Integer, String> typeNames = new LinkedHashMap<>(); // typeId -> name (from OpName)

			int wordOffset = 5; // skip header (5 words)
			int totalWords = spirv.remaining() / 4;

			while (wordOffset < totalWords) {
				int instructionWord = spirv.getInt(savedPos + wordOffset * 4);
				int opcode = instructionWord & 0xFFFF;
				int wordCount = (instructionWord >>> 16) & 0xFFFF;

				if (wordCount == 0) break; // safety

				// OpName (5): [opcode/wc] [id] [name_literal...]
				if (opcode == 5 && wordCount >= 3) {
					int id = spirv.getInt(savedPos + (wordOffset + 1) * 4);
					String name = readSPIRVString(spirv, savedPos + (wordOffset + 2) * 4, (wordCount - 2) * 4);
					typeNames.put(id, name);
				}

				// OpMemberName (6): [opcode/wc] [type_id] [member_idx] [name_literal...]
				if (opcode == 6 && wordCount >= 4) {
					int typeId = spirv.getInt(savedPos + (wordOffset + 1) * 4);
					int memberIdx = spirv.getInt(savedPos + (wordOffset + 2) * 4);
					String name = readSPIRVString(spirv, savedPos + (wordOffset + 3) * 4, (wordCount - 3) * 4);
					memberNames.computeIfAbsent(typeId, k -> new LinkedHashMap<>()).put(memberIdx, name);
				}

				// OpDecorate (71): [opcode/wc] [target_id] [decoration] [operands...]
				if (opcode == 71 && wordCount >= 3) {
					int targetId = spirv.getInt(savedPos + (wordOffset + 1) * 4);
					int decoration = spirv.getInt(savedPos + (wordOffset + 2) * 4);
					if (decoration == 2) { // Block decoration
						blockTypeIds.add(targetId);
					}
				}

				// OpMemberDecorate (72): [opcode/wc] [struct_type_id] [member_idx] [decoration] [operands...]
				if (opcode == 72 && wordCount >= 4) {
					int typeId = spirv.getInt(savedPos + (wordOffset + 1) * 4);
					int memberIdx = spirv.getInt(savedPos + (wordOffset + 2) * 4);
					int decoration = spirv.getInt(savedPos + (wordOffset + 3) * 4);

					if (decoration == 35 && wordCount >= 5) { // Offset
						int offset = spirv.getInt(savedPos + (wordOffset + 4) * 4);
						memberOffsets.computeIfAbsent(typeId, k -> new LinkedHashMap<>()).put(memberIdx, offset);
					}
					if (decoration == 4) { // RowMajor
						memberRowMajor.computeIfAbsent(typeId, k -> new LinkedHashMap<>()).put(memberIdx, true);
					}
					if (decoration == 5) { // ColMajor
						memberRowMajor.computeIfAbsent(typeId, k -> new LinkedHashMap<>()).put(memberIdx, false);
					}
				}

				wordOffset += wordCount;
			}

			// Find the IrisUniforms block type
			// Strategy: look for Block-decorated types that have member offsets
			Integer uboTypeId = null;
			for (Integer blockId : blockTypeIds) {
				if (memberOffsets.containsKey(blockId)) {
					// Check if this is our IrisUniforms block by looking at the type name
					String tName = typeNames.get(blockId);
					if (tName != null && tName.contains("IrisUniforms")) {
						uboTypeId = blockId;
						break;
					}
					// If no name match, take the first block type with offsets
					if (uboTypeId == null) {
						uboTypeId = blockId;
					}
				}
			}

			if (uboTypeId == null) {
				LOGGER.debug("[SPIRV_PARSE] No UBO block found in {} (blocks={}, offsets={})",
					shaderName, blockTypeIds.size(), memberOffsets.size());
				return null;
			}

			Map<Integer, String> names = memberNames.getOrDefault(uboTypeId, Map.of());
			Map<Integer, Integer> offsets = memberOffsets.getOrDefault(uboTypeId, Map.of());
			Map<Integer, Boolean> rowMajors = memberRowMajor.getOrDefault(uboTypeId, Map.of());

			SPIRVLayout layout = new SPIRVLayout(typeNames.getOrDefault(uboTypeId, "UBO"));
			for (Map.Entry<Integer, Integer> entry : offsets.entrySet()) {
				int memberIdx = entry.getKey();
				int offset = entry.getValue();
				String name = names.getOrDefault(memberIdx, "member_" + memberIdx);
				Boolean isRowMajor = rowMajors.get(memberIdx);
				layout.addMember(name, memberIdx, offset, isRowMajor);
			}

			return layout;

		} catch (Exception e) {
			LOGGER.warn("[SPIRV_PARSE] Failed to parse SPIR-V for {}: {}", shaderName, e.getMessage());
			return null;
		}
	}

	/**
	 * Reads a null-terminated UTF-8 string from SPIR-V literal string words.
	 */
	private static String readSPIRVString(ByteBuffer buf, int byteOffset, int maxBytes) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < maxBytes; i++) {
			byte b = buf.get(byteOffset + i);
			if (b == 0) break;
			sb.append((char) b);
		}
		return sb.toString();
	}

	/**
	 * Represents the parsed UBO layout from SPIR-V bytecode.
	 */
	public static class SPIRVLayout {
		public final String blockName;
		public final List<SPIRVMember> members = new ArrayList<>();

		public SPIRVLayout(String blockName) {
			this.blockName = blockName;
		}

		public void addMember(String name, int memberIndex, int byteOffset, Boolean isRowMajor) {
			members.add(new SPIRVMember(name, memberIndex, byteOffset, isRowMajor));
		}

		/** Get the byte offset for a named member, or -1 if not found */
		public int getOffset(String name) {
			for (SPIRVMember m : members) {
				if (m.name.equals(name)) return m.byteOffset;
			}
			return -1;
		}

		/** Check if any matrix member has RowMajor decoration */
		public boolean hasAnyRowMajor() {
			for (SPIRVMember m : members) {
				if (m.isRowMajor != null && m.isRowMajor) return true;
			}
			return false;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(String.format("SPIR-V UBO '%s' (%d members):\n", blockName, members.size()));
			for (SPIRVMember m : members) {
				sb.append(String.format("  [%4d] %-40s %s\n", m.byteOffset, m.name,
					m.isRowMajor != null ? (m.isRowMajor ? "RowMajor" : "ColMajor") : ""));
			}
			return sb.toString();
		}
	}

	public record SPIRVMember(String name, int memberIndex, int byteOffset, Boolean isRowMajor) {}
}
