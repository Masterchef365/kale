use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
use idek_basics::{
    idek::{self, nalgebra::Vector3},
    Array2D,
};
use rand::{distributions::Uniform, prelude::*};

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
        let z = 1.5;
        let sim = Simulation::new(w, w, z, rand::thread_rng(), |x, _y| x * 1.5 + 1.);

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
        let m = 1.5;

        self.sim.step(0.5 * m, 0.1 * m, 0.1 * m);

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
    pos: Vector3<f32>,
    dens: f32,
}

type Leaf = Array2D<Node>;

struct Simulation {
    /// The buffer presented next to the user, and read from during step
    front: Leaf,
    /// The buffer written to during a step
    back: Leaf,
}

impl Simulation {
    pub fn new(
        width: usize,
        height: usize,
        z_init: f32,
        mut rng: impl Rng,
        dens: fn(f32, f32) -> f32,
    ) -> Self {
        let z = Uniform::new(-z_init, z_init);

        let mut leaf = Leaf::new(width, height);
        for y in 0..leaf.height() {
            for x in 0..leaf.width() {
                leaf[(x, y)] = Node {
                    pos: Vector3::new(x as f32, z.sample(&mut rng), y as f32),
                    dens: dens(x as f32 / width as f32, y as f32 / height as f32),
                };
            }
        }

        Self {
            front: leaf.clone(),
            back: leaf,
        }
    }

    pub fn step(&mut self, spring: f32, restore: f32, square: f32) {
        for y in 0..self.front.height() {
            for x in 0..self.front.width() {
                let node = self.front[(x, y)];

                let offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)];

                let mut sum = Vector3::zeros();

                let mut middle = Vector3::zeros();

                let mut n_neighbors = 0;

                fn add_uv((x, y): (usize, usize), (ox, oy): (isize, isize)) -> (isize, isize) {
                    (ox + x as isize, oy + y as isize)
                }

                for off in offsets {
                    let grid_off = add_uv((x, y), off);
                    if let Some(pos) = self.front.bound(grid_off) {
                        let sample = self.front[pos];
                        let diff = node.pos - sample.pos;

                        let mag = diff.magnitude();

                        let n = diff.normalize();

                        middle += sample.pos;

                        sum += (node.dens - mag) * n;

                        n_neighbors += 1;
                    }
                }

                let middle = middle / n_neighbors as f32;

                let mut result = node;

                result.pos += sum * spring / n_neighbors as f32;

                if n_neighbors == 4 {
                    result.pos += restore * (middle - node.pos);
                }

                let pairs = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 0)];

                let mut n_pairs = 0;

                let mut avg_sq = Vector3::zeros();

                for pair in pairs.windows(2) {
                    let a = self.front.bound(add_uv((x, y), pair[0]));
                    let b = self.front.bound(add_uv((x, y), pair[1]));
                    if let Some((a, b)) = a.zip(b) {
                        let a = self.front[a].pos;
                        let b = self.front[b].pos;

                        let avg = (a + b) / 2.;
                        let diff = avg - node.pos;
                        let mag = diff.magnitude();
                        let r = (a - b).magnitude() / 2.;

                        avg_sq += diff.normalize() * (mag - r);

                        n_pairs += 1;
                    }
                }

                if n_pairs == 4 {
                    avg_sq /= n_pairs as f32;
                    result.pos += avg_sq * square;
                }

                self.back[(x, y)] = result;
            }
        }

        std::mem::swap(&mut self.front, &mut self.back);
    }

    fn data(&self) -> &Leaf {
        &self.front
    }
}

fn leaf_mesh(leaf: &Leaf, scale: f32) -> (Vec<Vertex>, Vec<u32>) {
    let max = leaf.width().max(leaf.height()) as f32;
    let vertices = leaf
        .data()
        .iter()
        .map(|node| {
            let pos = node.pos / max;
            Vertex::new(
                [
                    scale * (pos.x * 2. - 1.),
                    scale * pos.y,
                    scale * (pos.z * 2. - 1.),
                ],
                [node.dens - 1., 1., node.dens - 1.],
            )
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
