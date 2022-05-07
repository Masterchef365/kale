use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
use idek_basics::{
    idek::{self, nalgebra::Vector2},
    Array2D,
};

fn main() -> Result<()> {
    launch::<_, KaleApp>(Settings::default().vr_if_any_args())
}

struct KaleApp {
    verts: VertexBuffer,
    indices: IndexBuffer,
    sim: Simulation,
    shader: Shader,
    camera: MultiPlatformCamera,
}

const LEAF_SCALE: f32 = 3.;

impl App for KaleApp {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let w = 50;
        let sim = Simulation::new(w, w);

        let (vertices, indices) = leaf_mesh(sim.data(), LEAF_SCALE);

        Ok(Self {
            sim,
            verts: ctx.vertices(&vertices, true)?,
            indices: ctx.indices(&indices, false)?,
            shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Lines,
            )?,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        let (vertices, _) = leaf_mesh(self.sim.data(), LEAF_SCALE);

        ctx.update_vertices(self.verts, &vertices)?;

        Ok(vec![DrawCmd::new(self.verts)
            .indices(self.indices)
            .shader(self.shader)])
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

#[derive(Default, Copy, Clone)]
struct Node {
    pos: Vector2<f32>,
    dens: f32,
}

type Leaf = Array2D<Node>;

struct Simulation {
    leaf: Leaf,
}

impl Simulation {
    pub fn new(width: usize, height: usize) -> Self {
        let mut leaf = Leaf::new(width, height);
        for y in 0..leaf.height() {
            for x in 0..leaf.width() {
                leaf[(x, y)] = Node {
                    pos: Vector2::new(x as f32, y as f32),
                    dens: x as f32 / width as f32 + 1.,
                };
            }
        }

        Self { leaf }
    }

    fn data(&self) -> &Leaf {
        &self.leaf
    }
}

fn leaf_mesh(leaf: &Leaf, scale: f32) -> (Vec<Vertex>, Vec<u32>) {
    let max = leaf.width().max(leaf.height()) as f32;
    let vertices = leaf
        .data()
        .iter()
        .map(|node| {
            let pos = (node.pos / max) * 2. - Vector2::new(1., 1.);
            let pos = pos * scale;
            Vertex::new([pos.x, 0., pos.y], [node.dens - 1., 1., node.dens - 1.])
        })
        .collect();

    let mut indices = Vec::new();

    for y in 0..leaf.height() {
        for x in 0..leaf.width() {
            let center = leaf.calc_index((x, y));
            if x > 0 {
                let left = leaf.calc_index((x - 1, y));
                indices.push(center as u32);
                indices.push(left as u32);
            }

            if y > 0 {
                let up = leaf.calc_index((x, y - 1));
                indices.push(center as u32);
                indices.push(up as u32);
            }
        }
    }

    (vertices, indices)
}

fn neighbor_visitor<T>() {}