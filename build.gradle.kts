import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform

object Constants {
    // https://fabricmc.net/develop/
    const val MINECRAFT_VERSION: String = "1.21.1"
    const val FABRIC_LOADER_VERSION: String = "0.16.5"
    const val FABRIC_API_VERSION: String = "0.102.0+1.21.1"

    // https://semver.org/
    const val MOD_VERSION: String = "1.0.0-vulkan"

    const val IS_SHARED_BETA: Boolean = false
    const val ACTIVATE_RENDERDOC: Boolean = false
    const val BETA_TAG: String = "Vulkan Port"
    const val BETA_VERSION = 1

    // LWJGL versions (must match VulkanMod)
    const val LWJGL_VERSION: String = "3.4.1"
}

repositories {
    maven("https://maven.covers1624.net/")
}

plugins {
    // Unlike most projects, we choose to pin the specific version of Loom.
    // This prevents a lot of issues where the build script can fail randomly because the Fabric Maven server
    // is not reachable for some reason, and it makes builds much more reproducible. Observation also shows that it
    // really helps to improve startup times on slow connections.
    id("fabric-loom") version "1.6.5"
    id("com.github.gmazzo.buildconfig") version "5.3.5"
}

base {
    archivesName = "iris-vulkan"

    group = "net.irisshaders"
    version = "${Constants.MOD_VERSION}+mc${Constants.MINECRAFT_VERSION}"
}

loom {
    runs {
        create("clientWithQuickplay") {
            client()
            isIdeConfigGenerated = true
            programArgs("--launch_target", "net.fabricmc.loader.impl.launch.knot.KnotClient")
            mainClass.set("net.covers1624.devlogin.DevLogin")
            projectDir.resolve("run").resolve("mods").resolve(Constants.MINECRAFT_VERSION).mkdirs()
            vmArgs("-Dfabric.modsFolder=" + projectDir.resolve("run").resolve("mods").resolve(Constants.MINECRAFT_VERSION).absolutePath)
            programArgs("--quickPlaySingleplayer", "World For " + Constants.MINECRAFT_VERSION)
            if (Constants.ACTIVATE_RENDERDOC && DefaultNativePlatform.getCurrentOperatingSystem().isLinux) {
                environmentVariable("LD_PRELOAD", "/usr/lib/librenderdoc.so")
            }
        }
    }

    mixin {
        useLegacyMixinAp = false
    }

    accessWidenerPath = file("src/main/resources/iris.accesswidener")
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

sourceSets {
    val main = getByName("main")
    val test = getByName("test")
    val headers = create("headers")
    create("desktop")
    val vendored = create("vendored")

    headers.apply {
        java {
            compileClasspath += main.compileClasspath
        }
    }

    test.apply {
        java {
            compileClasspath += main.compileClasspath
            compileClasspath += main.output
        }
    }

    vendored.apply {
        java {
            compileClasspath += main.compileClasspath
        }
    }

    main.apply {
        java {
            compileClasspath += headers.output
            compileClasspath += vendored.output
            runtimeClasspath += vendored.output
        }
    }
}

buildConfig {
    className("BuildConfig")   // forces the class name. Defaults to 'BuildConfig'
    packageName("net.irisshaders.iris")  // forces the package. Defaults to '${project.group}'
    useJavaOutput()

    buildConfigField("IS_SHARED_BETA", Constants.IS_SHARED_BETA)
    buildConfigField("ACTIVATE_RENDERDOC", Constants.ACTIVATE_RENDERDOC)
    buildConfigField("BETA_TAG", Constants.BETA_TAG)
    buildConfigField("BETA_VERSION", Constants.BETA_VERSION)

    sourceSets.getByName("desktop") {
        buildConfigField("IS_SHARED_BETA", Constants.IS_SHARED_BETA)
    }
}


java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

dependencies {
    minecraft(group = "com.mojang", name = "minecraft", version = Constants.MINECRAFT_VERSION)
    mappings(loom.officialMojangMappings())
    modImplementation(group = "net.fabricmc", name = "fabric-loader", version = Constants.FABRIC_LOADER_VERSION)
    localRuntime("net.covers1624:DevLogin:0.1.0.5")

    // GLSL transformation pipeline (kept from original Iris - still needed for shader pack processing)
    include("org.antlr:antlr4-runtime:4.13.1")
    modImplementation("org.antlr:antlr4-runtime:4.13.1")
    include("io.github.douira:glsl-transformer:2.0.1")
    modImplementation("io.github.douira:glsl-transformer:2.0.1")
    include("org.anarres:jcpp:1.4.14")
    modImplementation("org.anarres:jcpp:1.4.14")

    // VulkanMod - compile against VulkanMod's classes (loaded as separate mod at runtime)
    // Build VulkanMod first: cd ../VulkanMod-1.21 && gradlew build
    // Then place the jar in the custom_vulkanmod/ directory
    modCompileOnly(fileTree("custom_vulkanmod") { include("*.jar") })

    // LWJGL Vulkan, VMA, and Shaderc (must match VulkanMod's versions)
    include(implementation("org.lwjgl:lwjgl-vulkan:${Constants.LWJGL_VERSION}")!!)

    fun includeLwjglNatives(name: String) {
        include(implementation("$name:${Constants.LWJGL_VERSION}")!!)
        include(runtimeOnly("$name:${Constants.LWJGL_VERSION}:natives-windows")!!)
        include(runtimeOnly("$name:${Constants.LWJGL_VERSION}:natives-linux")!!)
        include(runtimeOnly("$name:${Constants.LWJGL_VERSION}:natives-macos")!!)
        include(runtimeOnly("$name:${Constants.LWJGL_VERSION}:natives-macos-arm64")!!)
    }

    includeLwjglNatives("org.lwjgl:lwjgl-vma")
    includeLwjglNatives("org.lwjgl:lwjgl-shaderc")

    // MoltenVK for macOS Vulkan support
    include(runtimeOnly("org.lwjgl:lwjgl-vulkan:${Constants.LWJGL_VERSION}:natives-macos")!!)
    include(runtimeOnly("org.lwjgl:lwjgl-vulkan:${Constants.LWJGL_VERSION}:natives-macos-arm64")!!)

    // Distant Horizons API (optional compatibility)
    modCompileOnly(files(projectDir.resolve("DHApi.jar")))

    // Fabric API runtime dependencies
    modRuntimeOnly(fabricApi.module("fabric-rendering-fluids-v1", Constants.FABRIC_API_VERSION))
    modRuntimeOnly(fabricApi.module("fabric-resource-loader-v0", Constants.FABRIC_API_VERSION))

    fun addEmbeddedFabricModule(name: String) {
        val module = fabricApi.module(name, Constants.FABRIC_API_VERSION)
        modImplementation(module)
        include(module)
    }

    // Fabric API modules
    addEmbeddedFabricModule("fabric-api-base")
    addEmbeddedFabricModule("fabric-key-binding-api-v1")
    addEmbeddedFabricModule("fabric-rendering-v1")
}

tasks {
    runClient {
        if (Constants.ACTIVATE_RENDERDOC && DefaultNativePlatform.getCurrentOperatingSystem().isLinux) {
            environment("LD_PRELOAD", "/usr/lib/librenderdoc.so")
        }
        jvmArgs("-Dmixin.debug.export=true")
    }

    getByName<JavaCompile>("compileDesktopJava") {
        sourceCompatibility = JavaVersion.VERSION_1_8.toString()
        targetCompatibility = JavaVersion.VERSION_1_8.toString()
    }

    jar {
        from("${rootProject.projectDir}/LICENSE")
        duplicatesStrategy = DuplicatesStrategy.INCLUDE

        val vendored = sourceSets.getByName("vendored")
        from(vendored.output)

        val desktop = sourceSets.getByName("desktop")
        from(desktop.output)

        manifest.attributes["Main-Class"] = "net.irisshaders.iris.LaunchWarn"
    }

    processResources {
        inputs.property("version", project.version)

        filesMatching("fabric.mod.json") {
            expand(mapOf("version" to project.version))
        }
    }
}

// ensure that the encoding is set to UTF-8, no matter what the system default is
// this fixes some edge cases with special characters not displaying correctly
// see http://yodaconditions.net/blog/fix-for-java-file-encoding-problems-with-gradle.html
tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
}
