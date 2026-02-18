package net.irisshaders.iris.gl.uniform;

import org.lwjgl.system.MemoryUtil;

import java.nio.FloatBuffer;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Manages a native memory buffer for Iris shader uniform data (std140 layout).
 * Each ExtendedShader creates one of these to hold all non-opaque uniform values.
 * The buffer is uploaded to the GPU via VulkanMod's ManualUBO mechanism.
 *
 * Phase 7: Uniform System Adaptation - Vulkan Port.
 */
public class IrisUniformBuffer {
	private final long bufferPtr;
	private final int bufferSize;
	private final Map<String, FieldInfo> fields;
	private final int usedSize;
	private boolean transposeMatrices = false;

	public static class FieldInfo {
		public final String name;
		public final String type;
		public final int byteOffset;
		public final int byteSize;

		FieldInfo(String name, String type, int byteOffset, int byteSize) {
			this.name = name;
			this.type = type;
			this.byteOffset = byteOffset;
			this.byteSize = byteSize;
		}
	}

	public IrisUniformBuffer(Map<String, FieldInfo> fields, int usedSize) {
		this.bufferSize = Math.max(usedSize, 16);
		this.bufferPtr = MemoryUtil.nmemCalloc(1, bufferSize);
		this.fields = fields;
		this.usedSize = usedSize;
	}

	public long getPointer() { return bufferPtr; }
	public int getSize() { return bufferSize; }
	public int getUsedSize() { return usedSize; }
	public Map<String, FieldInfo> getFields() { return fields; }

	public int getFieldOffset(String name) {
		FieldInfo info = fields.get(name);
		return info != null ? info.byteOffset : -1;
	}

	// ==================== Read Methods ====================

	public float readFloat(int byteOffset) {
		if (byteOffset >= 0 && byteOffset + 4 <= bufferSize)
			return MemoryUtil.memGetFloat(bufferPtr + byteOffset);
		return 0.0f;
	}

	public int readInt(int byteOffset) {
		if (byteOffset >= 0 && byteOffset + 4 <= bufferSize)
			return MemoryUtil.memGetInt(bufferPtr + byteOffset);
		return 0;
	}

	// ==================== Write Methods ====================

	public void writeFloat(int byteOffset, float value) {
		if (byteOffset >= 0 && byteOffset + 4 <= bufferSize)
			MemoryUtil.memPutFloat(bufferPtr + byteOffset, value);
	}

	public void writeInt(int byteOffset, int value) {
		if (byteOffset >= 0 && byteOffset + 4 <= bufferSize)
			MemoryUtil.memPutInt(bufferPtr + byteOffset, value);
	}

	public void writeVec2f(int byteOffset, float x, float y) {
		if (byteOffset >= 0 && byteOffset + 8 <= bufferSize) {
			MemoryUtil.memPutFloat(bufferPtr + byteOffset, x);
			MemoryUtil.memPutFloat(bufferPtr + byteOffset + 4, y);
		}
	}

	public void writeVec2i(int byteOffset, int x, int y) {
		if (byteOffset >= 0 && byteOffset + 8 <= bufferSize) {
			MemoryUtil.memPutInt(bufferPtr + byteOffset, x);
			MemoryUtil.memPutInt(bufferPtr + byteOffset + 4, y);
		}
	}

	public void writeVec3f(int byteOffset, float x, float y, float z) {
		if (byteOffset >= 0 && byteOffset + 12 <= bufferSize) {
			MemoryUtil.memPutFloat(bufferPtr + byteOffset, x);
			MemoryUtil.memPutFloat(bufferPtr + byteOffset + 4, y);
			MemoryUtil.memPutFloat(bufferPtr + byteOffset + 8, z);
		}
	}

	public void writeVec3i(int byteOffset, int x, int y, int z) {
		if (byteOffset >= 0 && byteOffset + 12 <= bufferSize) {
			MemoryUtil.memPutInt(bufferPtr + byteOffset, x);
			MemoryUtil.memPutInt(bufferPtr + byteOffset + 4, y);
			MemoryUtil.memPutInt(bufferPtr + byteOffset + 8, z);
		}
	}

	public void writeVec4f(int byteOffset, float x, float y, float z, float w) {
		if (byteOffset >= 0 && byteOffset + 16 <= bufferSize) {
			MemoryUtil.memPutFloat(bufferPtr + byteOffset, x);
			MemoryUtil.memPutFloat(bufferPtr + byteOffset + 4, y);
			MemoryUtil.memPutFloat(bufferPtr + byteOffset + 8, z);
			MemoryUtil.memPutFloat(bufferPtr + byteOffset + 12, w);
		}
	}

	public void writeVec4i(int byteOffset, int x, int y, int z, int w) {
		if (byteOffset >= 0 && byteOffset + 16 <= bufferSize) {
			MemoryUtil.memPutInt(bufferPtr + byteOffset, x);
			MemoryUtil.memPutInt(bufferPtr + byteOffset + 4, y);
			MemoryUtil.memPutInt(bufferPtr + byteOffset + 8, z);
			MemoryUtil.memPutInt(bufferPtr + byteOffset + 12, w);
		}
	}

	/**
	 * Enable matrix transposition for this buffer. When true, writeMat4f/writeMat3f
	 * transpose from column-major (JOML/OpenGL) to row-major before writing.
	 *
	 * ExtendedShader (entity/hand/particle gbuffer shaders) needs this because
	 * shaderc-compiled SPIR-V reads UBO matrices transposed. Composite/final pass
	 * shaders do NOT need this (they work correctly without transpose).
	 */
	public void setTransposeMatrices(boolean transpose) {
		this.transposeMatrices = transpose;
	}

	public void writeMat4f(int byteOffset, FloatBuffer matrix) {
		if (byteOffset >= 0 && byteOffset + 64 <= bufferSize) {
			int pos = matrix.position();
			if (transposeMatrices) {
				for (int col = 0; col < 4; col++) {
					for (int row = 0; row < 4; row++) {
						MemoryUtil.memPutFloat(bufferPtr + byteOffset + (row * 4 + col) * 4,
							matrix.get(pos + col * 4 + row));
					}
				}
			} else {
				for (int i = 0; i < 16; i++) {
					MemoryUtil.memPutFloat(bufferPtr + byteOffset + i * 4, matrix.get(pos + i));
				}
			}
		}
	}

	public void writeMat4f(int byteOffset, float[] matrix) {
		if (byteOffset >= 0 && byteOffset + 64 <= bufferSize) {
			if (transposeMatrices) {
				for (int col = 0; col < 4; col++) {
					for (int row = 0; row < 4; row++) {
						int srcIdx = col * 4 + row;
						int dstIdx = row * 4 + col;
						if (srcIdx < matrix.length) {
							MemoryUtil.memPutFloat(bufferPtr + byteOffset + dstIdx * 4, matrix[srcIdx]);
						}
					}
				}
			} else {
				for (int i = 0; i < Math.min(16, matrix.length); i++) {
					MemoryUtil.memPutFloat(bufferPtr + byteOffset + i * 4, matrix[i]);
				}
			}
		}
	}

	/**
	 * Writes a mat3 in std140 layout: each column padded to vec4 (16 bytes).
	 * Total: 3 columns * 16 bytes = 48 bytes.
	 *
	 * When transposeMatrices is true, transposes column-major input to row-major:
	 *   Input:  [c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2]
	 *   Output: vec4[0]=row0, vec4[1]=row1, vec4[2]=row2
	 */
	public void writeMat3f(int byteOffset, float[] matrix) {
		if (byteOffset >= 0 && byteOffset + 48 <= bufferSize && matrix.length >= 9) {
			if (transposeMatrices) {
				// Transposed: each vec4 holds a row
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 0, matrix[0]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 4, matrix[3]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 8, matrix[6]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 16, matrix[1]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 20, matrix[4]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 24, matrix[7]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 32, matrix[2]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 36, matrix[5]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 40, matrix[8]);
			} else {
				// Column-major: each vec4 holds a column
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 0, matrix[0]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 4, matrix[1]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 8, matrix[2]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 16, matrix[3]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 20, matrix[4]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 24, matrix[5]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 32, matrix[6]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 36, matrix[7]);
				MemoryUtil.memPutFloat(bufferPtr + byteOffset + 40, matrix[8]);
			}
		}
	}

	/**
	 * Reads back a mat4 (16 floats, 64 bytes) from the buffer at the given offset.
	 * Returns null if out of bounds.
	 */
	public float[] readbackMat4f(int byteOffset) {
		if (byteOffset < 0 || byteOffset + 64 > bufferSize) return null;
		float[] result = new float[16];
		for (int i = 0; i < 16; i++) {
			result[i] = MemoryUtil.memGetFloat(bufferPtr + byteOffset + i * 4);
		}
		return result;
	}

	/**
	 * Reads back a vec3 (3 floats, 12 bytes) from the buffer at the given offset.
	 */
	public float[] readbackVec3f(int byteOffset) {
		if (byteOffset < 0 || byteOffset + 12 > bufferSize) return null;
		float[] result = new float[3];
		for (int i = 0; i < 3; i++) {
			result[i] = MemoryUtil.memGetFloat(bufferPtr + byteOffset + i * 4);
		}
		return result;
	}

	/**
	 * Reads back a single float (4 bytes) from the buffer at the given offset.
	 */
	public float[] readbackFloat(int byteOffset) {
		if (byteOffset < 0 || byteOffset + 4 > bufferSize) return null;
		return new float[] { MemoryUtil.memGetFloat(bufferPtr + byteOffset) };
	}

	/**
	 * Dumps the complete field map for diagnostic purposes.
	 * Returns a formatted string showing every field name, type, offset, and size.
	 */
	public String dumpFieldMap() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("IrisUniformBuffer: %d fields, usedSize=%d, bufferSize=%d, ptr=0x%X\n",
			fields.size(), usedSize, bufferSize, bufferPtr));
		// Sort by offset for readable output
		fields.values().stream()
			.sorted((a, b) -> Integer.compare(a.byteOffset, b.byteOffset))
			.forEach(f -> sb.append(String.format("  [%4d..%4d] %-8s %s\n",
				f.byteOffset, f.byteOffset + f.byteSize - 1, f.type, f.name)));
		return sb.toString();
	}

	/**
	 * Reads back raw bytes from the buffer for hex dump diagnostic.
	 */
	public byte[] readbackBytes(int byteOffset, int length) {
		if (byteOffset < 0 || byteOffset + length > bufferSize) return null;
		byte[] result = new byte[length];
		for (int i = 0; i < length; i++) {
			result[i] = MemoryUtil.memGetByte(bufferPtr + byteOffset + i);
		}
		return result;
	}

	public void free() {
		MemoryUtil.nmemFree(bufferPtr);
	}

	// ==================== Static Factory ====================

	private static final Pattern FIELD_PATTERN = Pattern.compile(
		"^\\s*(\\w+)\\s+(\\w+)(\\s*\\[[^]]*\\])?\\s*;");

	/**
	 * Parses the IrisUniforms UBO block from preprocessed Vulkan GLSL source
	 * and computes the std140 layout.
	 */
	public static IrisUniformBuffer fromVulkanGLSL(String vulkanSource) {
		Map<String, FieldInfo> fields = new LinkedHashMap<>();
		int currentOffset = 0;

		String[] lines = vulkanSource.split("\n");
		boolean inBlock = false;

		for (String line : lines) {
			String trimmed = line.trim();

			if (trimmed.contains("uniform IrisUniforms {")) {
				inBlock = true;
				continue;
			}

			if (inBlock && trimmed.equals("};")) {
				break;
			}

			if (inBlock) {
				Matcher m = FIELD_PATTERN.matcher(trimmed);
				if (m.find()) {
					String type = m.group(1);
					String name = m.group(2);
					String arrayPart = m.group(3);

					int arrayCount = 1;
					if (arrayPart != null && !arrayPart.isEmpty()) {
						try {
							arrayCount = Integer.parseInt(arrayPart.replaceAll("[\\[\\]\\s]", ""));
						} catch (NumberFormatException e) {
							arrayCount = 1;
						}
					}

					Std140Info info = getStd140Info(type);

					if (arrayCount <= 1) {
						// Non-array: align, record, advance
						currentOffset = alignTo(currentOffset, info.alignment);
						fields.put(name, new FieldInfo(name, type, currentOffset, info.size));
						currentOffset += info.size;
					} else {
						// Array: each element aligned to vec4 (16 bytes) in std140
						int elementAlignment = Math.max(info.alignment, 16);
						int elementSize = alignTo(info.size, 16);

						for (int a = 0; a < arrayCount; a++) {
							currentOffset = alignTo(currentOffset, elementAlignment);
							String fieldName = name + "[" + a + "]";
							fields.put(fieldName, new FieldInfo(fieldName, type, currentOffset, info.size));

							if (a == 0) {
								// Also map base name to first element
								fields.put(name, new FieldInfo(name, type, currentOffset, info.size));
							}
							currentOffset += elementSize;
						}
					}
				}
			}
		}

		// Align total size to 16 bytes (UBO alignment)
		currentOffset = alignTo(currentOffset, 16);

		return new IrisUniformBuffer(fields, currentOffset);
	}

	private static int alignTo(int offset, int alignment) {
		return (offset + alignment - 1) & ~(alignment - 1);
	}

	private static Std140Info getStd140Info(String type) {
		return switch (type) {
			case "float" -> new Std140Info(4, 4);
			case "int", "uint", "bool" -> new Std140Info(4, 4);
			case "vec2" -> new Std140Info(8, 8);
			case "ivec2", "uvec2", "bvec2" -> new Std140Info(8, 8);
			case "vec3" -> new Std140Info(16, 12);
			case "ivec3", "uvec3", "bvec3" -> new Std140Info(16, 12);
			case "vec4" -> new Std140Info(16, 16);
			case "ivec4", "uvec4", "bvec4" -> new Std140Info(16, 16);
			case "mat3" -> new Std140Info(16, 48); // 3 columns of vec4-padded vec3
			case "mat4" -> new Std140Info(16, 64); // 4 columns of vec4
			default -> new Std140Info(4, 4);
		};
	}

	private record Std140Info(int alignment, int size) {}
}
